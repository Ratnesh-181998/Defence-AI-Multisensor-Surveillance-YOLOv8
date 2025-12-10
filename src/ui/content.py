"""
Static content for the application documentation tabs.
"""

ABOUT_SECTIONS = {
    "intro": """
    ### 🛡️ Mission Dossier: Defence-Grade Vision System
    **Objective:** Design and deploy a real-time, multi-sensor AI surveillance system capable of operating in zero-visibility conditions with <500ms latency.
    """,
    "problem": """
    #### 🎯 Problem Statement
    To build a ruggedized AI system performing:
    *   **Dual-Spectrum Sights:** Object detection using both Day (Visible) and Thermal (LWIR) feeds.
    *   **Adverse Weather Ops:** Real-time fog/smoke removal.
    *   **Kinetic Tracking:** Calculating target speed and trajectory (Azimuth/Elevation).
    *   **Edge Supremacy:** Embedded inference on NVIDIA Jetson Orin AGX.
    """,
    "hardware": """
    #### ⚡ Hardware Configuration
    *   **Compute:** NVIDIA Jetson Orin AGX (Rugged Enclosure)
    *   **Vision Sensors:** 
        *   2x High-Res Day Cameras (GigE)
        *   2x LWIR Thermal Cameras (GigE)
    *   **Pipeline:** GStreamer-based low-latency architecture.
    """,
    "data": """
    #### 💾 Data & Annotation Strategy
    *   **Field Data:** Captured in fog, smoke, low-light, and dynamic environments.
    *   **Annotation:** Custom workflow using CVAT with specific Day/IR schemas.
    *   **Synthetic Ops:** Procedural generation of fog, noise, and thermal signatures for robust training.
    *   **Augmentation:** Mosaic, MixUp, and thermal-specific bloom/blur effects.
    """,
    "models": """
    #### 🧠 Model Engineering
    *   **Architectures:** YOLOv8m (Medium) & YOLOv8n (Nano) for speed/accuracy trade-off.
    *   **Benchmarking:** Detectron2 & Vision Transformers tested for long-range comparisons.
    *   **Optimization:** Mixed Precision (FP16) training and Class Balancing.
    """,
    "jetson": """
    #### 🚀 Jetson Orin Optimization
    *   **TensorRT Engine:** YOLO → ONNX → TensorRT conversion pipeline.
    *   **Acceleration:** Layer fusion and graph optimization achieved **~350–480 ms latency** across 4 streams.
    *   **Zero-Copy:** DeepStream integration for direct memory access (avoiding CPU bottlenecks).
    """,
    "tracking": """
    #### 📡 Tracking & Kinematics
    *   **Algorithm:** DeepSORT (Visual Feature Matching) + Kalman Filtering.
    *   **Physics Engine:** 
        *   Pixel-to-Degree mapping using camera intrinsics.
        *   Angular velocity estimation for moving targets.
        *   Target ID persistence during occlusion.
    """,
    "enhance": """
    #### 🌫️ Visibility Enhancement (Drishyak)
    *   **Techniques:** CLAHE (Contrast Limited Adaptive Histogram Equalization) & Dark Channel Prior.
    *   **Result:** 25-30 FPS video feed with digital smoke/fog removal.
    """,
    "fusion": """
    #### 🔗 Sensor Fusion
    *   **Alignment:** Time-synchronized streams with intrinsic/extrinsic calibration.
    *   **Logic:** Confidence-weighted voting (Thermal fallback when optical visibility is zero).
    """,
    "ui": """
    #### 🖥️ Operator Interface
    *   **Tech Stack:** Streamlit (Web) / PyQt.
    *   **Features:** Real-time "Glass Cockpit" with health telemetry (Temp, GPU Load), camera switching, and alerts.
    """,
    "docs": """
    #### 📄 Military-Grade Documentation
    *   **Standards:** DO-178C Inspired.
    *   **Deliverables:** SRS (Requirements), SDD (Design), STP (Test Plan), FTR (Field Trial Report).
    """,
    "trials": """
    #### ✅ Field Validation
    *   **Tested Environments:** Heavy fog, pitch darkness, high thermal noise.
    *   **Outcome:** Successfully validated for 24x7 stability and <500ms latency.
    """
}

HOW_IT_WORKS_STEPS = {
    "intro": """
    ### 🚀 How It Works: Step-by-Step Pipeline
    This system processes video frames through a sophisticated multi-stage pipeline, from raw sensor input to actionable intelligence on the operator's screen.
    """,
    "step1": {
        "title": "📹 Data Acquisition (The Eyes)",
        "icon": "📹",
        "content": """
        **The system ingests 4 synchronized video streams** via GigE (Gigabit Ethernet) interfaces:
        *   **2x Day Cameras** capture high-resolution color detail (1920x1080 @ 30 FPS)
        *   **2x Thermal Cameras** capture heat signatures (640x512 @ 30 FPS), crucial for detecting living targets or heated vehicles at night
        *   **Frame Synchronization:** Hardware-level timestamping ensures all 4 streams are aligned to within ±5ms
        """
    },
    "step2": {
        "title": "⚡ Pre-Processing & Enhancement (The Cornea)",
        "icon": "⚡",
        "content": """
        **Raw frames enter the pipeline** where they undergo critical transformations:
        *   **Resizing:** Frames normalized to 640x640 pixels (optimal for YOLO input)
        *   **Color Space Conversion:** BGR → RGB for day cameras, thermal mapping for IR
        *   **Visibility Enhancement (Drishyak Module):**
            *   **CLAHE** (Contrast Limited Adaptive Histogram Equalization) boosts detail in shadows
            *   **Dark Channel Prior** algorithm removes atmospheric fog/smoke digitally
            *   **Local Tone Mapping** preserves detail in both bright and dark regions
        """
    },
    "step3": {
        "title": "🧠 AI Inference (The Brain)",
        "icon": "🧠",
        "content": """
        **Enhanced frames are passed to TensorRT-optimized YOLOv8 engines:**
        *   **Parallel Inference:** Jetson Orin's GPU processes day and thermal feeds simultaneously using CUDA streams
        *   **TensorRT Acceleration:** Models quantized to FP16 (half-precision), achieving 3-5x speedup with <1% accuracy loss
        *   **Detection Classes:** `Person`, `Vehicle`, `Weapon`, `Animal` (custom-trained on 50K+ annotated images)
        *   **Output:** Bounding boxes with confidence scores (threshold: 0.5) and class labels
        """
    },
    "step4": {
        "title": "🎯 Tracking & Kinematics (The Memory)",
        "icon": "🎯",
        "content": """
        **Detections are passed to the DeepSORT Tracker:**
        *   **Kalman Filtering:** Predicts object position in next frame based on velocity/acceleration
        *   **Visual Feature Matching:** Extracts 128-dim appearance vectors using a CNN
        *   **ID Persistence:** Maintains track IDs even when targets are occluded for up to 30 frames
        *   **Kinematics Calculation:**
            *   **Pixel → Degree Mapping:** Converts bounding box center to azimuth/elevation angles
            *   **Angular Velocity:** Calculates target movement speed in degrees/second
            *   **Trajectory Prediction:** Estimates future position for next 5 seconds
        """
    },
    "step5": {
        "title": "🔗 Multi-Sensor Fusion (The Cortex)",
        "icon": "🔗",
        "content": """
        **Day and Thermal detections are fused using confidence-weighted voting:**
        *   **Spatial Alignment:** Detections within 50 pixels are considered the same target
        *   **Confidence Fusion:** Final confidence = 0.6 × Day_conf + 0.4 × Thermal_conf
        *   **Thermal Fallback:** If day camera visibility < 20% (fog/smoke), thermal takes priority
        *   **Cross-Validation:** Targets detected by both sensors marked as "High Confidence"
        """
    },
    "step6": {
        "title": "🖥️ Operator Display (The Interface)",
        "icon": "🖥️",
        "content": """
        **Final output is rendered in the Streamlit UI:**
        *   **Live Video Overlay:** Bounding boxes, track IDs, and confidence scores drawn on frames
        *   **System Telemetry:** Real-time CPU/GPU/Memory usage, temperature monitoring
        *   **Alert System:** Audio/visual alerts for high-priority detections (weapons, intrusions)
        *   **Data Logging:** All detections saved to SQLite database with timestamps
        *   **Performance:** Entire pipeline runs at 25-30 FPS with <500ms end-to-end latency
        """
    }
}

# Keep the old string for backward compatibility (will be deprecated)
HOW_IT_WORKS = """
### 🚀 How It Works: Step-by-Step

#### 1. Data Acquisition (The Eyes)
*   The system ingests **4 synchronized video streams** via GigE (Gigabit Ethernet) interfaces.
*   **2x Day Cameras** capture high-resolution color detail.
*   **2x Thermal Cameras** capture heat signatures, crucial for detecting living targets or heated vehicles at night.

#### 2. Pre-Processing & Enhancement (The Cornea)
*   Raw frames enter the pipeline where they are resized and normalized.
*   **Visibility Enhancement:** If fog or smoke is detected, the **CLAHE (Contrast Limited Adaptive Histogram Equalization)** and Dark Channel Prior algorithms activate to "dehaze" the image digitally before it even reaches the AI.

#### 3. AI Inference (The Brain)
*   The enhanced frames are passed to **TensorRT-optimized YOLOv8 engines**.
*   **Parallel Inference:** The Jetson Orin's GPU processes day and thermal feeds simultaneously.
*   **TensorRT Acceleration:** Models are quantised to FP16 (half-precision), speeding up calculations by 3x without losing accuracy.

#### 4. Tracking & Kinematics (The Memory)
*   Detections are passed to the **DeepSORT Tracker**.
*   **Association:** The system matches new detections to existing tracks using Kalman Filtering (predicting where a target *should* be).
*   **ID Persistence:** If a target walks behind a tree, the system remembers its ID for a set duration.
*   **Kinematics:** The system calculates angular velocity and estimated distance based on pixel movement relative to camera intrinsics.

#### 5. Operator Display (The Interface)
*   The final output overlays bounding boxes, confidence scores, and track IDs onto the video.
*   Operators view this in the **Streamlit UI**, which also provides system health telemetry (temperature, GPU load) to ensure hardware reliability.
"""

ARCHITECTURE_DIAGRAM = """
### 🏗️ System Architecture

The system follows a modular **Producer-Consumer** pipeline design to ensure zero-latency bottlenecks.

```mermaid
graph TD
    subgraph Sensors
        C1[Day Cam 1] -->|GigE Stream| Q1(Input Queue)
        C2[Day Cam 2] -->|GigE Stream| Q1
        T1[Thermal 1] -->|GigE Stream| Q2(Input Queue)
        T2[Thermal 2] -->|GigE Stream| Q2
    end

    subgraph "Edge Processor (Jetson Orin)"
        Q1 & Q2 --> PRE[Pre-Processing]
        PRE -->|Dehazing| VIS[Drishyak Enhancement]
        VIS -->|Batched Frames| TRT[TensorRT Engine]
        
        TRT -->|Detections| SORT[DeepSORT Tracker]
        SORT -->|Track IDs| KIN[Kinematics Logic]
    end

    subgraph "Operator Station"
        KIN --> UI[Streamlit Interface]
        UI --> DIS[Live Display]
        UI --> LOG[Event Logs]
    end
    
    classDef hardware fill:#f9f,stroke:#333;
    classDef logic fill:#bbf,stroke:#333;
    class Sensors,hardware;
    class TRT,SORT,logic;
```

```

### 🧠 Step-by-Step System Flow

This architecture is designed to handle massive data throughput with minimal latency. Here is exactly what happens to every single video frame, from the camera lens to the operator's screen:

#### 1. 📷 Sensor Data Acquisition (The Inputs)
*   **Physical Sensors:** The system starts with 4 ruggedized cameras. Two **Day Cameras** provide standard color video, while two **Thermal (LWIR)** cameras detect heat signatures.
*   **GigE Interface:** Cameras are connected via Gigabit Ethernet, ensuring high bandwidth (1000 Mbps) transmission.
*   **Frame Queueing:** To prevent the system from crashing if the AI gets busy, incoming frames are placed in a **Thread-Safe Queue**. This acts as a buffer zone.

#### 2. ⚡ Pre-Processing & Synchronization
*   **Normalization:** Raw video data is often in different formats. The pre-processor converts everything into a standard mathematical matrix (Tensor).
*   **Time-Sync:** The logic ensures that a frame from the Day Camera and a frame from the Thermal Camera taken at the exact same millisecond are processed together.
*   **Resizing:** Images are resized to **640x640** pixels, the optimal size for our AI models to understand.

#### 3. 🌫️ Visibility Enhancement (The 'Drishyak' Module)
*   *Before* the AI looks at the image, we clean it.
*   **Fog Removal:** Using the **Dark Channel Prior** algorithm, we estimate how much "fog" is in the air and mathematically subtract it.
*   **Contrast Boosting:** **CLAHE** (Contrast Limited Adaptive Histogram Equalization) is applied to maintain detail in both very bright and very dark areas (like shadows).

#### 4. 🧠 AI Inference Engine (The Brain)
*   **TensorRT Optimization:** We don't just run standard Python code. The AI models (YOLOv8) are compiled into **TensorRT Engines**. This compresses the neural network, making it run 3-5x faster on NVIDIA hardware.
*   **Mixed Precision (FP16):** We assume "half-precision" math (16-bit floating point). This reduces memory usage by 50% with almost zero loss in accuracy.
*   **Detection:** The AI scans the image for trained classes: `Soldiers`, `Vehicles`, `Weapons`, etc.

#### 5. 🎯 DeepSORT Tracking ( The Memory)
*   **Problem:** YOLO only sees "individual frames". It doesn't know that the person in Frame 1 is the same person in Frame 2.
*   **Solution (DeepSORT):**
    *   **Kalman Filtering:** Based on the object's speed in previous frames, the system *predicts* where it will be next.
    *   **Visual Matching:** It extracts a "feature vector" (appearance signature) of the object. If a target goes behind a tree and re-appears, the system recognizes its signature and restores the original ID.

#### 6. 📐 Kinematics & Fusion
*   **3D Estimation:** Using camera parameters (focal length), we convert 2D pixel coordinates into estimated real-world angles (Azimuth/Elevation).
*   **Fusion:** If the Thermal camera sees a target but the Day camera is blocked by smoke, the system fuses this data to confirm a valid detection, reducing false alarms.

#### 7. 🖥️ Operator Dashboard (The Output)
*   The final processed frames, along with health data (CPU temp, Memory usage), are sent to the Streamlit UI.
*   This creates a "Glass Cockpit" experience for the operator, giving them total situational awareness in real-time.
"""

TECH_STACK = """
### 🛠️ Technology Arsenel

<div style="margin-bottom: 20px;">
    <h4 style="color: #667eea; margin-bottom: 10px;">⚡ Edge Hardware</h4>
    <span class="stack-badge">🖥️ NVIDIA Jetson Orin AGX (64GB)</span>
    <span class="stack-badge">🐢 275 TOPS AI Compute</span>
    <span class="stack-badge">📹 Teledyne FLIR GigE</span>
    <span class="stack-badge">🔌 PoE Network Switch</span>
</div>

<div style="margin-bottom: 20px;">
    <h4 style="color: #667eea; margin-bottom: 10px;">🧠 AI Core</h4>
    <span class="stack-badge">🔥 PyTorch (Training)</span>
    <span class="stack-badge">🚀 NVIDIA TensorRT (Inference)</span>
    <span class="stack-badge">👁️ YOLOv8 (Custom Trained)</span>
    <span class="stack-badge">🎯 DeepSORT Tracking</span>
    <span class="stack-badge">🔮 Kalman Filters</span>
</div>

<div style="margin-bottom: 20px;">
    <h4 style="color: #667eea; margin-bottom: 10px;">👁️ Vision Pipeline</h4>
    <span class="stack-badge">🐍 Python 3.8+</span>
    <span class="stack-badge">🖼️ OpenCV (Image Proc)</span>
    <span class="stack-badge">🔢 NumPy (Matrix Ops)</span>
    <span class="stack-badge">✨ CLAHE Enhancement</span>
    <span class="stack-badge">🌫️ Dark Channel Prior</span>
</div>

<div>
    <h4 style="color: #667eea; margin-bottom: 10px;">🚢 Deployment</h4>
    <span class="stack-badge">🐳 Docker Containers</span>
    <span class="stack-badge">🎨 Streamlit Custom UI</span>
    <span class="stack-badge">🐙 Git Version Control</span>
    <span class="stack-badge">🔧 CUDA Acceleration</span>
</div>
"""
