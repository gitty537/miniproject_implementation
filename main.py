"""
PPE Detection Backend  (ONNX Runtime)
--------------------------------------
Run:  uvicorn main:app --reload
Open: http://localhost:8000
"""

import asyncio
import base64
import json
import time
import sqlite3
import shutil
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# ── ONNX Detector ───────────────────────────────────────────────────────────
CLASS_NAMES = {0: "person", 1: "helmet", 2: "vest"}

# Per-class confidence thresholds
# helmet/vest are lower so we avoid false *violation* alerts
CONF_THRESHOLDS = {
    "person": 0.35,
    "helmet": 0.30,
    "vest":   0.30,
}

# NMS parameters
NMS_SCORE_THRESH = 0.25   # hard pre-filter before NMS
NMS_IOU_THRESH   = 0.50   # suppress overlapping boxes of same class

# Containment threshold for PPE → person assignment
CONTAIN_THRESH = 0.50     # ≥50% of equipment box must be inside person box

# Temporal smoothing — how many consecutive frames without gear = real violation
# At 15 FPS this means ~1 second of consistently missing gear before flagging
VIOLATION_FRAMES = 15
# IoU threshold for re-identifying the same person across consecutive frames
TRACK_IOU_THRESH = 0.30


class ONNXDetector:
    """Wraps best.onnx and exposes a simple detect(frame) → list[dict] API."""

    INPUT_SIZE = 640   # model expects 640×640

    def __init__(self, model_path: str):
        # Use CPU provider for broad compatibility
        self.sess = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name  = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        print(f"[ONNX] Loaded model from {model_path}")

    # ── Pre-processing ──────────────────────────────────────────────────────
    def _letterbox(self, img):
        """Resize with letterbox padding to INPUT_SIZE × INPUT_SIZE."""
        h, w = img.shape[:2]
        scale = self.INPUT_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        pad_top  = (self.INPUT_SIZE - new_h) // 2
        pad_left = (self.INPUT_SIZE - new_w) // 2

        canvas = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return canvas, scale, pad_top, pad_left

    def _preprocess(self, frame):
        img, scale, pad_top, pad_left = self._letterbox(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float16) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]   # [1, 3, 640, 640]
        return img, scale, pad_top, pad_left

    # ── Post-processing ─────────────────────────────────────────────────────
    def _postprocess(self, raw, orig_h, orig_w, scale, pad_top, pad_left):
        """
        raw shape: [1, 7, 8400]
        Each anchor: [cx, cy, w, h, score_person, score_helmet, score_vest]
        """
        preds = raw[0].T   # [8400, 7]  (float16 → float32 for safe maths)
        preds = preds.astype(np.float32)

        boxes_cx  = preds[:, 0]
        boxes_cy  = preds[:, 1]
        boxes_w   = preds[:, 2]
        boxes_h   = preds[:, 3]
        class_scores = preds[:, 4:]   # [8400, 3]

        class_ids   = np.argmax(class_scores, axis=1)     # [8400]
        confidences = class_scores[np.arange(len(preds)), class_ids]  # [8400]

        # Convert cx,cy,w,h → x1,y1,x2,y2 (in letterbox space)
        x1 = boxes_cx - boxes_w / 2
        y1 = boxes_cy - boxes_h / 2
        x2 = boxes_cx + boxes_w / 2
        y2 = boxes_cy + boxes_h / 2

        detections = []
        # Run NMS per class
        for cls_id in range(len(CLASS_NAMES)):
            label = CLASS_NAMES[cls_id]
            cls_thresh = CONF_THRESHOLDS[label]

            mask = (class_ids == cls_id) & (confidences >= max(NMS_SCORE_THRESH, cls_thresh))
            if not mask.any():
                continue

            cls_boxes  = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1)
            cls_scores = confidences[mask]

            # cv2.dnn.NMSBoxes wants [x, y, w, h] in a list
            nms_input = [[float(b[0]), float(b[1]),
                          float(b[2] - b[0]), float(b[3] - b[1])]
                         for b in cls_boxes]
            keep = cv2.dnn.NMSBoxes(
                nms_input,
                cls_scores.tolist(),
                cls_thresh,
                NMS_IOU_THRESH,
            )
            if keep is None or len(keep) == 0:
                continue

            keep = keep.flatten()
            for idx in keep:
                bx1, by1, bx2, by2 = cls_boxes[idx]

                # Undo letterbox padding and scale
                bx1 = (bx1 - pad_left) / scale
                by1 = (by1 - pad_top)  / scale
                bx2 = (bx2 - pad_left) / scale
                by2 = (by2 - pad_top)  / scale

                # Clip to frame bounds
                bx1 = int(max(0, bx1))
                by1 = int(max(0, by1))
                bx2 = int(min(orig_w - 1, bx2))
                by2 = int(min(orig_h - 1, by2))

                if bx2 <= bx1 or by2 <= by1:
                    continue

                detections.append({
                    "box":        [bx1, by1, bx2, by2],
                    "class_id":   cls_id,
                    "label":      label,
                    "confidence": round(float(cls_scores[idx]), 2),
                })

        return detections

    # ── Public API ──────────────────────────────────────────────────────────
    def detect(self, frame) -> list:
        orig_h, orig_w = frame.shape[:2]
        tensor, scale, pad_top, pad_left = self._preprocess(frame)
        raw = self.sess.run([self.output_name], {self.input_name: tensor})
        return self._postprocess(raw[0], orig_h, orig_w, scale, pad_top, pad_left)


detector = ONNXDetector("best.onnx")
# ────────────────────────────────────────────────────────────────────────────


# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_PATH   = "sample.mp4"
TARGET_FPS   = 15
JPEG_QUALITY = 75

COLOURS = {
    "person":    (200, 200, 200),
    "helmet":    (50,  200,  50),
    "vest":      (50,  200, 200),
    "VIOLATION": (30,   30, 220),
}
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="PPE Detection")

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Database ─────────────────────────────────────────────────────────────────
db_conn = sqlite3.connect("alerts.db", check_same_thread=False)
db_conn.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        frame     INTEGER,
        video_time REAL,
        message   TEXT
    )
""")
db_conn.commit()


# ── Shared state ─────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.running       = False
        self.paused        = False
        self.video_path    = VIDEO_PATH
        self.frame_count   = 0
        self.total_frames  = 0
        self.violations    = 0
        self.recent_alerts = deque(maxlen=50)
        self.clients: list[WebSocket] = []

state = AppState()


# ── Person Tracker ────────────────────────────────────────────────────────────
class TrackedPerson:
    """Remembers how many consecutive frames a person has been missing gear."""
    _next_id = 0

    def __init__(self, box):
        self.id                 = TrackedPerson._next_id
        TrackedPerson._next_id += 1
        self.box                = box
        self.no_helmet_streak   = 0   # consecutive frames missing helmet
        self.no_vest_streak     = 0   # consecutive frames missing vest
        self.age                = 0   # total frames this track has been alive
        self.violation_alerted  = False  # have we already inserted this to DB?


class PersonTracker:
    """
    Frame-to-frame simple IoU tracker for person boxes.
    Each call to update() matches new detections to existing tracks.
    """
    def __init__(self):
        self.tracks: list[TrackedPerson] = []

    def _box_iou(self, a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, person_detections: list) -> list[TrackedPerson]:
        """
        Match current-frame person boxes to existing tracks.
        Returns the list of TrackedPerson objects in same order as input.
        """
        new_boxes = [d["box"] for d in person_detections]
        matched_tracks = [None] * len(new_boxes)
        used_tracks = set()

        # Greedy nearest-match by IoU
        for i, box in enumerate(new_boxes):
            best_iou, best_t = 0.0, -1
            for t_idx, track in enumerate(self.tracks):
                if t_idx in used_tracks:
                    continue
                iou_val = self._box_iou(box, track.box)
                if iou_val > best_iou:
                    best_iou, best_t = iou_val, t_idx
            if best_iou >= TRACK_IOU_THRESH:
                matched_tracks[i] = self.tracks[best_t]
                used_tracks.add(best_t)

        # Create new tracks for unmatched detections
        result = []
        for i, box in enumerate(new_boxes):
            if matched_tracks[i] is None:
                matched_tracks[i] = TrackedPerson(box)
            track = matched_tracks[i]
            track.box = box   # update position
            track.age += 1
            result.append(track)

        # Keep only tracks that were matched this frame
        self.tracks = result
        return result

    def reset(self):
        self.tracks = []
        TrackedPerson._next_id = 0


tracker = PersonTracker()


# ── PPE Compliance Logic ──────────────────────────────────────────────────────
def _containment(person_box, equip_box) -> float:
    """
    Fraction of the equipment box that lies inside the person box.
    Returns a value in [0, 1].
    """
    px1, py1, px2, py2 = person_box
    ex1, ey1, ex2, ey2 = equip_box

    ix1 = max(px1, ex1);  iy1 = max(py1, ey1)
    ix2 = min(px2, ex2);  iy2 = min(py2, ey2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    equip_area = (ex2 - ex1) * (ey2 - ey1)
    return inter / equip_area if equip_area > 0 else 0.0


def analyse_detections(detections: list) -> tuple[list, list]:
    """
    For each person, check whether a helmet and vest are contained inside
    their bounding box using containment scoring.

    Temporal smoothing: only flag a violation after VIOLATION_FRAMES
    consecutive frames of missing equipment, suppressing single-frame flickers.
    """
    persons = [d for d in detections if d["label"] == "person"]
    helmets = [d for d in detections if d["label"] == "helmet"]
    vests   = [d for d in detections if d["label"] == "vest"]

    # Greedy assignment: each piece of equipment to the person whose box
    # contains the most of the equipment box
    def assign(equipment_list, persons):
        assigned_to = {}
        for e_idx, equip in enumerate(equipment_list):
            best_score, best_p = 0.0, -1
            for p_idx, person in enumerate(persons):
                score = _containment(person["box"], equip["box"])
                if score > best_score:
                    best_score, best_p = score, p_idx
            if best_score >= CONTAIN_THRESH:
                assigned_to[e_idx] = best_p
        return set(assigned_to.values())

    covered_by_helmet = assign(helmets, persons)
    covered_by_vest   = assign(vests,   persons)

    # Match persons to tracked identities
    tracked = tracker.update(persons)

    violations = []
    for p_idx, (p, track) in enumerate(zip(persons, tracked)):
        missing_helmet = p_idx not in covered_by_helmet
        missing_vest   = p_idx not in covered_by_vest

        # Update streaks
        track.no_helmet_streak = track.no_helmet_streak + 1 if missing_helmet else 0
        track.no_vest_streak   = track.no_vest_streak   + 1 if missing_vest   else 0

        # Only report as a real violation after N consecutive bad frames
        real_missing = []
        if track.no_helmet_streak >= VIOLATION_FRAMES:
            real_missing.append("no helmet")
        if track.no_vest_streak >= VIOLATION_FRAMES:
            real_missing.append("no vest")

        if real_missing:
            p["violation"] = True
            p["missing"]   = real_missing
            violations.append({
                "frame": state.frame_count,
                "time":  round(state.frame_count / max(TARGET_FPS, 1), 1),
                "msg":   ", ".join(real_missing).capitalize(),
            })
        else:
            p["violation"] = False

    return detections, violations


# ── Frame drawing ────────────────────────────────────────────────────────────
def draw_frame(frame: np.ndarray, detections: list) -> np.ndarray:
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label   = d["label"]
        conf    = d.get("confidence", 0)
        is_viol = d.get("violation", False)

        colour = COLOURS["VIOLATION"] if is_viol else COLOURS.get(label, (180, 180, 180))

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        tag = f"{label} {conf:.0%}"
        if is_viol:
            missing = " · ".join(d.get("missing", []))
            tag = f"! {missing.upper()}"

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(frame, tag, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame,
                f"Frame {state.frame_count}  |  Violations: {state.violations}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ── Video loop ────────────────────────────────────────────────────────────────
async def video_loop():
    tracker.reset()   # clear stale person tracks from any previous session
    path_or_idx = (int(state.video_path)
                   if str(state.video_path).isdigit()
                   else state.video_path)

    cap = cv2.VideoCapture(path_or_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {state.video_path}")
        state.running = False
        return

    state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_native         = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval     = 1.0 / TARGET_FPS
    state.frame_count  = 0

    print(f"[INFO] Opened {state.video_path}  |  "
          f"{state.total_frames} frames  |  native {fps_native:.1f} fps")

    while state.running:
        if state.paused:
            await asyncio.sleep(0.1)
            continue

        t0 = time.monotonic()

        ret, frame = cap.read()
        if not ret:
            # Video file finished or webcam disconnected — stop
            break

        state.frame_count += 1

        # ── Detect ──
        raw_detections = detector.detect(frame)

        # ── Compliance ──
        detections, new_violations = analyse_detections(raw_detections)
        if new_violations:
            state.violations += len(new_violations)
            for v in new_violations:
                state.recent_alerts.appendleft(v)
                db_conn.execute(
                    "INSERT INTO alerts (frame, video_time, message) VALUES (?, ?, ?)",
                    (v["frame"], v["time"], v["msg"]),
                )
            db_conn.commit()

        # ── Draw & encode ──
        annotated = draw_frame(frame.copy(), detections)
        _, buf = cv2.imencode(".jpg", annotated,
                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        b64 = base64.b64encode(buf).decode()

        msg = json.dumps({
            "type":       "frame",
            "image":      b64,
            "frame":      state.frame_count,
            "total":      state.total_frames,
            "violations": state.violations,
            "alerts":     list(state.recent_alerts)[:10],
            "detections": [
                {k: v for k, v in d.items() if k != "box"}
                for d in detections
            ],
        })

        dead = []
        for ws in state.clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            state.clients.remove(ws)

        elapsed = time.monotonic() - t0
        await asyncio.sleep(max(0, frame_interval - elapsed))

    cap.release()
    print("[INFO] Video loop stopped.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("templates/index.html").read_text(encoding="utf-8")


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = f"uploaded_{file.filename}"
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    return {"path": file_path}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.clients.append(ws)
    print(f"[WS] Client connected  ({len(state.clients)} total)")
    try:
        while True:
            data = await ws.receive_text()
            cmd  = json.loads(data)

            if cmd.get("action") == "start":
                if not state.running:
                    if cmd.get("path"):
                        state.video_path = cmd["path"]
                    state.running    = True
                    state.paused     = False
                    state.violations = 0
                    state.recent_alerts.clear()
                    asyncio.create_task(video_loop())
                    print("[INFO] Detection started.")

            elif cmd.get("action") == "pause":
                state.paused = not state.paused

            elif cmd.get("action") == "stop":
                state.running = False
                state.paused  = False
                print("[INFO] Detection stopped.")

            elif cmd.get("action") == "set_path":
                state.video_path = cmd["path"]

    except WebSocketDisconnect:
        state.clients.remove(ws)
        print(f"[WS] Client disconnected  ({len(state.clients)} total)")
