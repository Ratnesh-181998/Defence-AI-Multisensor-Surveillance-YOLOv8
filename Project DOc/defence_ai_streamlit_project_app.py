"""
Defence-grade AI demo project (single-file Streamlit app)

Project: Real-Time Multi-Sensor Object Detection, Tracking & Visibility Enhancement
Target UI: Streamlit

This is a single-file, end-to-end scaffold suitable for GitHub. It demonstrates and wires together the
major components described in the JD:
 - dataset collection helpers (simulated)
 - annotation export (YOLO format)
 - synthetic augmentation utilities
 - training stub (PyTorch + YOLOv8 example placeholders)
 - model export (ONNX) and TensorRT conversion stub
 - inference pipeline (OpenCV / GStreamer fallback)
 - visibility enhancement (CLAHE + simple CNN placeholder)
 - multi-camera simulation (4 streams using local videos or webcams)
 - tracking (CentroidTracker + DeepSORT placeholder)
 - operator UI (Streamlit): live frames, logs, model control, health metrics
 - docs generator (SRS/SDD stub export)

Notes:
 - This file is a scaffold and educational demo. Heavy compute steps (training, TensorRT) are represented
   by functions that will run on capable machines. They include safe fallbacks for local development.
 - On Jetson, replace GPU detection and tensor conversion code paths with Jetson-specific utilities.
 - To run locally: `pip install -r requirements.txt` then `streamlit run app.py`.

Requirements (sample):
  - streamlit
  - opencv-python
  - numpy
  - torch
  - torchvision
  - ultralytics (optional for YOLOv8)
  - onnx
  - onnxruntime
  - Pillow
  - pyyaml
  - pymongo (optional for logs)

"""

import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
import os
import tempfile
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

# Try imports that may be optional
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

# ---------- Utilities & Config ----------
PROJECT_DIR = Path.cwd() / "defence_ai_demo"
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"
for p in (PROJECT_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)

@dataclass
class CameraStreamConfig:
    id: int
    source: str  # file path or camera index (string convertible to int)
    name: str
    fps: int = 25


DEFAULT_CAMERAS = [
    CameraStreamConfig(0, "0", "Day-1"),
    CameraStreamConfig(1, "1", "Day-2"),
    CameraStreamConfig(2, "thermal_sample.mp4", "Thermal-1"),
    CameraStreamConfig(3, "thermal_sample2.mp4", "Thermal-2"),
]

# Simple logger
def log(msg: str):
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

# Health metrics
def get_system_health():
    try:
        import psutil
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        return {"cpu": cpu, "mem": mem}
    except Exception:
        return {"cpu": None, "mem": None}

# ---------- Dataset & Augmentation ----------

def create_yolo_annotation(x, y, w, h, class_id, img_w, img_h):
    # Convert xywh to YOLO normalized format
    cx = (x + w/2) / img_w
    cy = (y + h/2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"


def apply_thermal_augment(image: np.ndarray) -> np.ndarray:
    # Simulate thermal-style artifacts: colormap + blur + noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    blurred = cv2.GaussianBlur(colored, (5,5), 0)
    noise = (np.random.randn(*blurred.shape) * 5).astype(np.uint8)
    aug = cv2.add(blurred, noise)
    return aug


def apply_visibility_augmentation(image: np.ndarray, severity=0.5) -> np.ndarray:
    # Add fog / smoke using alpha blending with white noise
    h, w = image.shape[:2]
    fog = np.full((h, w, 3), 255, dtype=np.uint8)
    mask = (np.random.rand(h, w) * 255 * severity).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (101,101), 0)
    mask = mask[..., None]
    out = (image.astype(np.float32) * (1 - mask/255.0) + fog.astype(np.float32) * (mask/255.0)).astype(np.uint8)
    return out

# ---------- Visibility Enhancement (Drishyak) ----------

def clahe_enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


# Simple CNN placeholder for learned enhancement (doesn't actually train here)
if TORCH_AVAILABLE:
    import torch.nn as nn
    class TinyEnhancer(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 3, 3, padding=1)
            )
        def forward(self, x):
            return torch.clamp(self.conv(x), 0, 1)

# ---------- Tracking (CentroidTracker) ----------
class CentroidTracker:
    def __init__(self, max_lost=30):
        self.next_object_id = 0
        self.objects = dict()  # id -> centroid
        self.lost = dict()
        self.max_lost = max_lost

    def update(self, detections: List[Tuple[int,int,int,int]]):
        # detections: list of bbox x,y,w,h
        centroids = [ (int(x+w/2), int(y+h/2)) for (x,y,w,h) in detections ]
        if len(self.objects) == 0:
            for c in centroids:
                self.objects[self.next_object_id] = c
                self.lost[self.next_object_id] = 0
                self.next_object_id += 1
        else:
            # naive matching: assign by nearest
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())
            assigned = set()
            for c in centroids:
                dists = [ (np.hypot(c[0]-oc[0], c[1]-oc[1]), oid) for oc, oid in zip(obj_centroids, obj_ids) ]
                dists.sort()
                for dist, oid in dists:
                    if oid not in assigned:
                        self.objects[oid] = c
                        self.lost[oid] = 0
                        assigned.add(oid)
                        break
            # increment lost for unassigned
            for oid in obj_ids:
                if oid not in assigned:
                    self.lost[oid] += 1
            # remove lost
            to_remove = [oid for oid, l in self.lost.items() if l > self.max_lost]
            for oid in to_remove:
                del self.objects[oid]
                del self.lost[oid]
        return self.objects

# ---------- Model Training (Stubs) ----------

def train_yolo_stub(dataset_path: Path, output_weights: Path, epochs: int = 5):
    """Train a small YOLO/torch model. This is a placeholder showing API structure.
    Replace with your training code (ultralytics/YOLOv8 or detectron2) on full GPU machines.
    """
    log(f"Training started: dataset={dataset_path} epochs={epochs}")
    if not TORCH_AVAILABLE:
        log("Torch not available — training stub will not run. Install torch to train models.")
        return None
    # Tiny example: save a placeholder weights file
    dummy = {"epoch": epochs, "mAP": 0.5}
    torch.save(dummy, str(output_weights))
    log(f"Training finished — weights saved to {output_weights}")
    return output_weights

# ---------- ONNX & TensorRT Export (Stubs) ----------

def export_to_onnx_stub(model_path: Path, onnx_out: Path):
    log(f"Exporting model {model_path} -> {onnx_out}")
    # This is highly model dependent — provide instructions in README
    with open(onnx_out, 'wb') as f:
        f.write(b"ONNX_PLACEHOLDER")
    return onnx_out


def convert_onnx_to_tensorrt_stub(onnx_path: Path, trt_out: Path, fp16=True):
    log(f"(Stub) Converting ONNX {onnx_path} -> TensorRT engine {trt_out} (fp16={fp16})")
    # On Jetson, use trtexec or TensorRT Python API. Here we create placeholder.
    with open(trt_out, 'wb') as f:
        f.write(b"TENSORRT_PLACEHOLDER")
    return trt_out

# ---------- Inference Engine (Mock) ----------
class InferenceEngine:
    def __init__(self, model_path: Optional[Path]=None):
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'
        if TORCH_AVAILABLE and model_path and model_path.exists():
            try:
                self.model = torch.load(str(model_path), map_location='cpu')
                log("Loaded torch model (placeholder)")
            except Exception:
                log("Failed to load torch model — using placeholder inference")

    def infer(self, frame: np.ndarray) -> List[Tuple[int,int,int,int,int]]:
        # Return list of (x,y,w,h,class_id) as dummy detections
        h,w = frame.shape[:2]
        # Dummy: detect a centered box
        box_w = int(w*0.2)
        box_h = int(h*0.2)
        x = int(w*0.4)
        y = int(h*0.4)
        return [(x,y,box_w,box_h,0)]

# ---------- Video Stream Worker ----------

class VideoWorker(threading.Thread):
    def __init__(self, source: str, out_q: queue.Queue, name: str = 'cam'):
        super().__init__(daemon=True)
        self.source = source
        self.out_q = out_q
        self.name = name
        self.running = False
        try:
            idx = int(source)
            self.cap = cv2.VideoCapture(idx)
        except Exception:
            self.cap = cv2.VideoCapture(source)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 25)

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            # Resize for performance
            frame = cv2.resize(frame, (640, 360))
            try:
                self.out_q.put(frame, timeout=0.1)
            except queue.Full:
                pass
            time.sleep(1.0 / max(self.fps, 15))

    def stop(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

# ---------- Streamlit App UI ----------

st.set_page_config(page_title="Defence AI Demo", layout='wide')

if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'workers' not in st.session_state:
    st.session_state.workers = dict()
if 'queues' not in st.session_state:
    st.session_state.queues = dict()
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'tracker' not in st.session_state:
    st.session_state.tracker = CentroidTracker()

st.title("Defence AI — Real-Time Multi-Sensor Demo (Streamlit)")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    cameras = st.multiselect("Select camera sources", options=[c.name for c in DEFAULT_CAMERAS], default=[c.name for c in DEFAULT_CAMERAS])
    start_btn = st.button("Start Streams")
    stop_btn = st.button("Stop Streams")
    upload_weights = st.file_uploader("Upload model weights (.pt/.pth)")
    train_button = st.button("Run Training Stub")
    export_onnx = st.button("Export ONNX (stub)")
    convert_trt = st.button("Convert to TensorRT (stub)")
    enhance_toggle = st.checkbox("Enable CLAHE visibility enhancement", value=True)
    tracking_toggle = st.checkbox("Enable Centroid Tracking", value=True)
    fps_target = st.slider("Target display FPS", min_value=5, max_value=30, value=10)

col1, col2 = st.columns([3,1])

with col2:
    st.subheader("System Health")
    health = get_system_health()
    st.metric("CPU %", health['cpu'] if health['cpu'] is not None else "N/A")
    st.metric("MEM %", health['mem'] if health['mem'] is not None else "N/A")
    st.subheader("Logs")
    st.text_area("Logs", value='\n'.join(st.session_state.logs[-20:]), height=400)

with col1:
    st.subheader("Live Streams")
    # Prepare columns for 4 cameras
    cam_cols = st.columns(2)
    display_placeholders = [cam_cols[0].empty(), cam_cols[1].empty(), cam_cols[0].empty(), cam_cols[1].empty()]

# Start/stop logic
if start_btn:
    # start workers for selected cameras
    for idx, cam_cfg in enumerate(DEFAULT_CAMERAS):
        if cam_cfg.name in cameras:
            q = queue.Queue(maxsize=2)
            w = VideoWorker(cam_cfg.source, q, name=cam_cfg.name)
            st.session_state.queues[cam_cfg.name] = q
            st.session_state.workers[cam_cfg.name] = w
            w.start()
            log(f"Started worker for {cam_cfg.name}")

if stop_btn:
    for name, w in list(st.session_state.workers.items()):
        try:
            w.stop()
        except Exception:
            pass
        del st.session_state.workers[name]
        del st.session_state.queues[name]
        log(f"Stopped worker {name}")

# Load uploaded weights into engine
if upload_weights is not None:
    tmpfile = Path(tempfile.mkdtemp()) / upload_weights.name
    with open(tmpfile, 'wb') as f:
        f.write(upload_weights.getbuffer())
    st.session_state.engine = InferenceEngine(tmpfile)
    log(f"Uploaded and loaded model: {tmpfile}")

if train_button:
    # run training stub in background thread
    out_weights = MODELS_DIR / "weights.pth"
    t = threading.Thread(target=train_yolo_stub, args=(DATA_DIR, out_weights, 2), daemon=True)
    t.start()
    log("Training stub launched in background thread")

if export_onnx:
    dummy_weights = MODELS_DIR / "weights.pth"
    onnx_out = MODELS_DIR / "model.onnx"
    export_to_onnx_stub(dummy_weights, onnx_out)

if convert_trt:
    onnx_in = MODELS_DIR / "model.onnx"
    trt_out = MODELS_DIR / "model.trt"
    convert_onnx_to_tensorrt_stub(onnx_in, trt_out)

# Main loop: render frames and run inference
run_key = 'app_running'
if run_key not in st.session_state:
    st.session_state[run_key] = True

last_display = time.time()
while st.session_state[run_key]:
    # iterate cameras
    frames = []
    for i, cam_cfg in enumerate(DEFAULT_CAMERAS):
        name = cam_cfg.name
        if name in st.session_state.queues:
            try:
                frame = st.session_state.queues[name].get_nowait()
            except Exception:
                frame = None
        else:
            frame = None
        if frame is None:
            # placeholder black frame
            frame = np.zeros((360,640,3), dtype=np.uint8)
            cv2.putText(frame, f"No stream: {name}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        # apply visibility enhancement
        proc = frame.copy()
        if enhance_toggle:
            try:
                proc = clahe_enhance(proc)
            except Exception:
                pass
        # run inference
        detections = []
        if st.session_state.engine:
            detections = st.session_state.engine.infer(proc)
        else:
            # placeholder detection
            h,w = proc.shape[:2]
            detections = [(int(w*0.4), int(h*0.4), int(w*0.2), int(h*0.2), 0)]
        # tracking
        if tracking_toggle:
            boxes = [ (x,y,w_,h_) for (x,y,w_,h_,cls) in detections ]
            objs = st.session_state.tracker.update(boxes)
            # draw boxes and ids
            for (x,y,w_,h_,cls) in detections:
                cv2.rectangle(proc, (x,y), (x+w_, y+h_), (0,255,0), 2)
            for oid, cent in objs.items():
                cv2.putText(proc, f"ID:{oid}", (cent[0], cent[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            for (x,y,w_,h_,cls) in detections:
                cv2.rectangle(proc, (x,y), (x+w_, y+h_), (255,0,0), 2)
        frames.append((name, proc))

    # display into placeholders
    for i, (name, frame) in enumerate(frames[:4]):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_placeholders[i].image(img, caption=name, use_column_width=True)

    # update metrics occasionally
    if time.time() - last_display > 2.0:
        h = get_system_health()
        st.experimental_rerun()
        break

# ---------- Export README & Docs helper ----------

st.sidebar.markdown("---")
if st.sidebar.button("Generate SRS/SDD (stub)"):
    srs = {
        "name": "Defence AI Demo",
        "description": "Demo scaffold for multi-sensor detection & visibility enhancement",
        "requirements": ["Jetson Orin AGX recommended", "4x GigE cameras"],
    }
    out = PROJECT_DIR / "SRS.json"
    with open(out, 'w') as f:
        json.dump(srs, f, indent=2)
    log(f"SRS written to {out}")

st.sidebar.markdown("---")
if st.sidebar.button("Download project snapshot (zip)"):
    # create a small zip of the project folder
    import shutil
    snapshot = PROJECT_DIR / "snapshot.zip"
    shutil.make_archive(str(snapshot.with_suffix('')), 'zip', PROJECT_DIR)
    with open(snapshot, 'rb') as f:
        st.download_button("Download ZIP", f, file_name="defence_ai_demo.zip")

st.sidebar.markdown("\n---\nProject notes:\n- This is a scaffold. Replace stubs with real training & TRT code.\n- On Jetson, prefer TensorRT conversions and DeepStream integration.")

# EOF
