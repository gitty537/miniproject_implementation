# PPE Detection — Local Backend

Real-time PPE violation detection on .mp4 video, streamed to a browser dashboard.

---

## Folder structure

```
ppe_backend/
├── main.py              ← FastAPI server (edit this to plug in your model)
├── requirements.txt
├── templates/
│   └── index.html       ← Browser dashboard
└── static/              ← Put any extra CSS/JS here
```

---

## Setup (one time)

```bash
cd ppe_backend
pip install -r requirements.txt
```

---

## Run

```bash
uvicorn main:app --reload
```

Then open **http://localhost:8000** in your browser.

---

## Plug in YOUR model

Open `main.py` and find this section near the top:

```python
# ── Import your model here ──────────────────────────
class MockModel:   ← DELETE this whole class
    ...

model = MockModel()   ← REPLACE with your real model
```

**If your model uses Ultralytics (standard YOLOv8/MFD-YOLO):**

```python
from ultralytics import YOLO

model_raw = YOLO("path/to/your/mfd_yolo_1_5.pt")

def model(frame):
    results = model_raw(frame, verbose=False)[0]
    detections = []
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
```

**If your model returns raw tensors**, wrap them to produce the same dict format:
```python
{
    "box":        [x1, y1, x2, y2],   # pixel coords, integers
    "label":      "person",            # "person", "helmet", or "vest"
    "confidence": 0.87,               # float 0-1
    "class_id":   0,                  # int
}
```

The rest of the pipeline (violation logic, drawing, WebSocket streaming) works
automatically from that format.

---

## Tuning

In `main.py`, near the top:

| Variable        | Default | What it does                              |
|-----------------|---------|-------------------------------------------|
| `VIDEO_PATH`    | `sample.mp4` | Default video file path             |
| `TARGET_FPS`    | `15`    | Frames sent to browser per second         |
| `JPEG_QUALITY`  | `75`    | Compression (lower = faster, blurrier)    |
| `CONF_THRESH`   | `0.40`  | Ignore detections below this confidence   |
| `MIN_IOU`       | `0.05`  | Overlap needed to match PPE to person     |

---

## How violation detection works

1. Model returns bounding boxes for `person`, `helmet`, `vest`
2. For every `person` box, check if any `helmet` / `vest` box overlaps it
3. If a person has no overlapping helmet → **violation: no helmet**
4. If a person has no overlapping vest   → **violation: no vest**
5. Violation is drawn in red on the frame and logged to the alert panel

---

## Switch to a real IP camera (RTSP) later

Change `VIDEO_PATH` to your camera URL:
```python
VIDEO_PATH = "rtsp://admin:password@192.168.1.100:554/stream"
```
No other code changes needed.
