# 🔄 Application Execution Flow: Step-by-Step

This document explains exactly how the **Defence AI System** runs, from the moment you hit "Enter" to processing live video feeds.

---

## 🟢 PHASE 1: Application Startup
**Command Triggered:** `streamlit run app.py`

1.  **`app.py` loads**: Python reads the file from top to bottom.
2.  **Imports & Config**:
    -   Libraries loaded (`streamlit`, `cv2`, `torch`, `ultralytics`).
    -   `st.set_page_config` sets the title and wide layout.
    -   Custom CSS is injected (`apply_custom_css()`) for the dark theme.
3.  **Session State Initialization (`init_session_state()`)**:
    -   Streamlit re-runs the script on every interaction. To keep data alive, we use `st.session_state`.
    -   **Critical Variables created**:
        -   `stream_active = False` (System is off)
        -   `workers = {}` (No cameras running)
        -   `engine = None` (AI model not loaded yet)
        -   `logs = []` (Log history empty)

---

## 🟡 PHASE 2: The Main Interface Loop
Streamlit draws the UI from top to bottom every time you click something.

1.  **Sidebar (`render_sidebar()`)**:
    -   Checks system health (CPU/GPU/RAM).
    -   Shows status indicators.
2.  **Tabs Created**: The main view is split into tabs (`About`, `Control Panel`, `Live Streams`, etc.).
3.  **Control Panel Render**:
    -   Shows Camera Selection (Signal Sources).
    -   Shows "ENGAGE" and "ABORT" buttons.
    -   Shows Model Config.

---

## 🔴 PHASE 3: "ENGAGE" - System Activation
**User Action:** Clicks `▶️ ENGAGE` button in Control Panel.

1.  **Button Handler Triggered**:
    -   `st.session_state.stream_active` set to `True`.
    -   **`inference_engine.py`** is initialized (Loads YOLOv8 model into memory).
2.  **Camera Workers Started**:
    -   The app checks which cameras are checked (e.g., Day-1, Thermal-1).
    -   For each active camera, a **`VideoWorker`** thread is spawned (`src/camera/video_worker.py`).
    -   **Thread Action**:
        -   Each worker opens its video source (`cv2.VideoCapture`).
        -   It starts reading frames in a continuous `while` loop.
        -   It pushes frames into a thread-safe **Queue**.

---

## 🟣 PHASE 4: The Core Processing Loop
This is the "heartbeat" of the system, running continuously while active.

1.  **Frame Retrieval**:
    -   The main app asks the queues: *"Do you have a new frame?"*
2.  **Pipeline Processing (`inference_engine.py`)**:
    For every frame:
    -   **Step A: Enhancement** (if enabled):
        -   Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
        -   Uses Dark Channel Prior to remove fog/smoke.
    -   **Step B: Detection (YOLOv8)**:
        -   Model scans image -> Returns Bounding Boxes (xyxy), Classes, Confidence.
    -   **Step C: Tracking (DeepSORT)**:
        -   Takes detections and assigns unique IDs (Experiment #1 -> ID: 42).
        -   Predicts next position using Kalman Filters.
    -   **Step D: Fusion** (if enabled):
        -   Matches Day Camera boxes with Thermal Camera boxes.
        -   Combines them for higher confidence.
3.  **Visualization**:
    -   Draws bounding boxes and labels on the frame.
    -   Updates counters (Persons: 5, Vehicles: 2).

---

## 🔵 PHASE 5: Display & Update
1.  **Streamlit Display**:
    -   The processed frame is converted to RGB.
    -   `st.image()` updates the placeholder in the **Live Streams** tab.
2.  **Metrics Update**:
    -   FPS (Frames Per Second) is calculated.
    -   CPU/GPU usage updated in the sidebar.
3.  **Loop**:
    -   `st.rerun()` is called (or a loop inside the tab continues) to process the next frame immediately.

---

## 🟤 PHASE 6: Training Flow (When "Train Model" is clicked)
1.  **Validation**: Checks if `data/data.yaml` exists.
2.  **Subprocess Launch**:
    -   Runs `python src/training/train_yolo.py` as a separate background process.
    -   It does NOT block the main UI (you can still click around).
3.  **Training Script**:
    -   Loads YOLOv8n.pt (Pre-trained weights).
    -   Reads your images (dataset).
    -   Optimizes weights (Backpropagation).
    -   Saves `best.pt` to `models/trained/`.

---

## ℹ️ Summary of Key Files

| File | Purpose |
|------|---------|
| `app.py` | **The Brain**. Handles UI, user state, and coordinates everything. |
| `src/camera/video_worker.py` | **The Eyes**. Reads video frames in background threads. |
| `src/detection/inference_engine.py` | **The Cortex**. Runs YOLO detection on frames. |
| `src/tracking/deep_sort.py` | **The Memory**. Tracks objects over time. |
| `src/enhancement/clahe.py` | **The Glasses**. Cleans up foggy/dark images. |
