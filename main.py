"""
PPE Detection Backend
---------------------
Run:  uvicorn main:app --reload
Open: http://localhost:8000
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil

# ── Import your model here ──────────────────────────────────────────────────
from ultralytics import YOLO
import sys
import types

# Create a mock mmcv module to use the project's Conv2d fallback
sys.modules['mmcv'] = types.ModuleType('mmcv')
sys.modules['mmcv.ops'] = types.ModuleType('mmcv.ops')

import torch.nn as nn
class MockModulatedDeformConv2d(nn.Conv2d):
    def forward(self, x, offset=None, mask=None):
        return super().forward(x)

sys.modules['mmcv.ops'].ModulatedDeformConv2d = MockModulatedDeformConv2d
sys.modules['mmcv.ops.modulated_deform_conv'] = types.ModuleType('mmcv.ops.modulated_deform_conv')
sys.modules['mmcv.ops.modulated_deform_conv'].ModulatedDeformConv2dPack = MockModulatedDeformConv2d
sys.modules['mmcv.ops.modulated_deform_conv'].ModulatedDeformConv2d = MockModulatedDeformConv2d

import mfd_yolo

import torch

# Monkey patch torch.load to bypass weights_only restriction in PyTorch 2.6+
_original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

model_raw = YOLO("best.pt")

def model(frame):
    results = model_raw(frame, verbose=False)[0]
    detections = []
    if results.boxes is None:
        return detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = results.names[cls_id]
        detections.append({
            "box":        [x1, y1, x2, y2],
            "class_id":   cls_id,
            "label":      label,
            "confidence": round(conf, 2),
        })
    return detections
# ────────────────────────────────────────────────────────────────────────────


# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_PATH   = "sample.mp4"   # ← change to your .mp4 path
TARGET_FPS   = 15             # frames sent to browser per second
JPEG_QUALITY = 75             # 1-100, lower = faster / smaller
CONF_THRESH  = 0.40           # ignore detections below this confidence

# Colours per class (BGR for OpenCV)
COLOURS = {
    "person":  (200, 200, 200),
    "helmet":  (50,  200, 50),
    "vest":    (50,  200, 200),
    "VIOLATION": (30, 30, 220),
}
# ────────────────────────────────────────────────────────────────────────────


app = FastAPI(title="PPE Detection")

# Serve the static folder (JS/CSS if you add any)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Shared state and DB ──────────────────────────────────────────────────────
import sqlite3

db_conn = sqlite3.connect("alerts.db", check_same_thread=False)
db_conn.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        frame INTEGER,
        video_time REAL,
        message TEXT
    )
""")
db_conn.commit()

class AppState:
    def __init__(self):
        self.running       = False
        self.paused        = False
        self.video_path    = VIDEO_PATH
        self.frame_count   = 0
        self.total_frames  = 0
        self.violations    = 0          # running total
        self.recent_alerts = deque(maxlen=50)
        self.clients: list[WebSocket] = []

state = AppState()


# ── PPE violation logic ──────────────────────────────────────────────────────
def analyse_detections(detections: list) -> tuple[list, list]:
    """
    Returns (annotated_detections, violations).

    Rule: a 'person' box that has no 'helmet' or 'vest' box overlapping
    it by at least MIN_IOU is flagged as a violation.
    """
    MIN_IOU = 0.05   # low threshold — equipment boxes are smaller than person

    persons  = [d for d in detections if d["label"] == "person"]
    helmets  = [d for d in detections if d["label"] == "helmet"]
    vests    = [d for d in detections if d["label"] == "vest"]

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        # Calculate intersection over the equipment's area (area B)
        # This ensures small helmets inside large persons get a high score
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / area_b if area_b > 0 else 0.0

    violations = []
    for p in persons:
        has_helmet = any(iou(p["box"], h["box"]) > MIN_IOU for h in helmets)
        has_vest   = any(iou(p["box"], v["box"]) > MIN_IOU for v in vests)
        missing = []
        if not has_helmet: missing.append("no helmet")
        if not has_vest:   missing.append("no vest")
        if missing:
            p["violation"] = True
            p["missing"]   = missing
            violations.append({
                "frame": state.frame_count,
                "time":  round(state.frame_count / max(TARGET_FPS, 1), 1),
                "msg":   ", ".join(missing).capitalize(),
            })
        else:
            p["violation"] = False

    return detections, violations


# ── Frame drawing ────────────────────────────────────────────────────────────
def draw_frame(frame: np.ndarray, detections: list) -> np.ndarray:
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label    = d["label"]
        conf     = d.get("confidence", 0)
        is_viol  = d.get("violation", False)

        colour = COLOURS["VIOLATION"] if is_viol else COLOURS.get(label, (180, 180, 180))
        thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

        tag = f'{label} {conf:.0%}'
        if is_viol:
            missing = " · ".join(d.get("missing", []))
            tag = f'⚠ {missing.upper()}'

        # label background
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(frame, tag, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Frame counter overlay
    cv2.putText(frame,
                f'Frame {state.frame_count}  |  Violations: {state.violations}',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ── Video loop (runs as background asyncio task) ────────────────────────────
async def video_loop():
    # If path is exclusively digits, treat it as webcam index
    path_or_idx = int(state.video_path) if str(state.video_path).isdigit() else state.video_path
    
    cap = cv2.VideoCapture(path_or_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {state.video_path}")
        state.running = False
        return

    state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_native         = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval     = 1.0 / TARGET_FPS
    state.frame_count  = 0

    print(f"[INFO] Opened {state.video_path}  |  {state.total_frames} frames  |  native {fps_native:.1f} fps")

    while state.running:
        if state.paused:
            await asyncio.sleep(0.1)
            continue

        t0 = time.monotonic()

        ret, frame = cap.read()
        if not ret:
            # Loop the video if it's a file
            if state.total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state.frame_count = 0
                continue
            else:
                # Disconnected from webcam stream
                break

        state.frame_count += 1

        # ── Run detection ──
        raw_detections = [
            d for d in model(frame)
            if d.get("confidence", 1) >= CONF_THRESH
        ]

        # ── Analyse PPE compliance ──
        detections, new_violations = analyse_detections(raw_detections)
        if new_violations:
            state.violations += len(new_violations)
            for v in new_violations:
                state.recent_alerts.appendleft(v)
                # Store in DB
                db_conn.execute("INSERT INTO alerts (frame, video_time, message) VALUES (?, ?, ?)", 
                                (v["frame"], v["time"], v["msg"]))
            db_conn.commit()

        # ── Draw annotations ──
        annotated = draw_frame(frame.copy(), detections)

        # ── Encode to JPEG ──
        _, buf = cv2.imencode(".jpg", annotated,
                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        b64 = base64.b64encode(buf).decode()

        # ── Build message ──
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

        # ── Broadcast to all connected browser tabs ──
        dead = []
        for ws in state.clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            state.clients.remove(ws)

        # ── Throttle to TARGET_FPS ──
        elapsed = time.monotonic() - t0
        await asyncio.sleep(max(0, frame_interval - elapsed))

    cap.release()
    print("[INFO] Video loop stopped.")


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path("templates/index.html")).read_text(encoding="utf-8")


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = f"uploaded_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"path": file_path}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.clients.append(ws)
    print(f"[WS] Client connected  ({len(state.clients)} total)")
    try:
        while True:
            data = await ws.receive_text()
            cmd = json.loads(data)

            if cmd.get("action") == "start":
                if not state.running:
                    if cmd.get("path"):
                        state.video_path = cmd["path"]
                    state.running = True
                    state.paused  = False
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
