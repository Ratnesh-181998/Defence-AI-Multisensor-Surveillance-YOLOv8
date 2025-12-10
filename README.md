# 🛡️ Defence AI: Multi-Sensor Surveillance System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-SOTA-green)](https://github.com/ultralytics/ultralytics)
[![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Orin-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Real-time, defence-grade AI system for multi-sensor object detection, tracking, and visibility enhancement on NVIDIA Jetson Orin AGX.**


<img width="1278" height="1075" alt="image" src="https://github.com/user-attachments/assets/7e210829-4c29-4d6d-874c-489715505992" />
<img width="1276" height="1285" alt="image" src="https://github.com/user-attachments/assets/db767b3e-6007-4206-a22a-3fa890682f4c" />
<img width="1272" height="950" alt="image" src="https://github.com/user-attachments/assets/e2cd0122-9e7e-4ff2-a15d-07f6e2fbfda6" />

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Hardware Requirements](#-hardware-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Deployment](#-deployment)
- [Performance Metrics](#-performance-metrics)
- [Contact](#-contact)
- [License](#-license)

---

## 🎯 Overview

**Defence AI Multisensor Surveillance** is a cutting-edge computer vision platform designed for mission-critical environmental monitoring and threat detection. It integrates **Day and Thermal (LWIR) camera feeds** to provide 24/7 situational awareness, utilizing state-of-the-art **YOLOv8** for detection and **DeepSORT** for robust tracking.

Optimized for the **NVIDIA Jetson Orin AGX**, this system delivers real-time inference (<500ms latency) even in degraded visual environments (fog, smoke, low light) thanks to its proprietary **Drishyak** visibility enhancement module.

<img width="2806" height="1440" alt="image" src="https://github.com/user-attachments/assets/89d56afd-bd65-4da3-ab3a-b0fb497b5682" />
<img width="2794" height="1445" alt="image" src="https://github.com/user-attachments/assets/39527480-322a-48b9-b32b-2b76f4aeb572" />
<img width="2827" height="1441" alt="image" src="https://github.com/user-attachments/assets/005bf75d-7828-4d25-8821-7be7f8054d8b" />
<img width="2811" height="1311" alt="image" src="https://github.com/user-attachments/assets/d702d0cf-c6c7-4487-ad4f-c686678fe399" />


### 🎥 Demo Capability
*   **Sensor Fusion:** Simultaneous processing of multiple GigE streams.
*   **Edge AI:** Full on-device processing with no cloud dependency.
*   **Tactical Dashboard:** Interactive Streamlit UI for operators.

---
## 🌐🎬 Live Demo
🚀 **Try it now:**
- **Streamlit Profile** - https://share.streamlit.io/user/ratnesh-181998
- **Project Demo** - https://defence-ai-multisensor-surveillance-yolov8-vusybzt9bohpykhzkq3.streamlit.app/
- **Frontend**: Streamlit + Custom CSS
- **Detection**: YOLOv8 + PyTorch + TensorRT
- **Tracking**: DeepSORT + Kalman Filter
- **Processing**: OpenCV + NumPy
- **Enhancement**: CLAHE + Dark Channel Prior
- **Hardware**: NVIDIA Jetson Orin AGX
---
## ✨ Key Features

### 🔍 Detection & Tracking
- **Multi-Spectral Detection:** Seamlessly detects targets in RGB and Thermal spectrums using custom-trained YOLOv8 models.
- **Robust Tracking:** Implements DeepSORT with Kalman filtering for consistent ID retention despite occlusions.
- **Kinematics:** Estimates target azimuth, elevation, and velocity vectors.

### 🌫️ Visibility Enhancement (Drishyak)
- **CLAHE Optimization:** Contrast Limited Adaptive Histogram Equalization for detail recovery.
- **Dehazing:** Dark Channel Prior algorithms to neutralize atmospheric scattering (fog/smoke).
- **Auto-Switching:** Intelligent pipeline that activates enhancement based on scene analysis.

### ⚡ Performance Engineering
- **TensorRT Acceleration:** FP16 precision optimization for 3-5x inference speedup on Jetson.
- **Zero-Copy Pipeline:** Efficient memory management for high-throughput video processing.
- **Asynchronous Design:** Non-blocking capture and inference threads.

### 🖥️ Operator Interface
- **Command & Control:** Centralized dashboard for system health, camera control, and recording.
- **Analytics Suite:** Real-time metrics, historical data analysis, and PDF/CSV reporting.
- **Event Logging:** Comprehensive logging of all system detections and user actions.

---

## 🖥️ User Interface Experience

The application features a professional, tab-based command center designed for ease of use by defence operators. Below is a detailed breakdown of each interface module:

### 1. ⚙️ Control Panel (Command Center)
**Functionality:**
The "Heart" of the system. Operators use this tab to configure signal sources (Webcam vs Simulation), manage the processing sequence (ENGAGE/ABORT), and load AI models.
- **Signal Sources:** Toggle individual camera feeds (Day/Thermal).
- **Sequence Control:** One-click system activation with visual status indicators.
- **AI Core:** Drag-and-drop interface to load custom YOLOv8 `.pt` or TensorRT `.trt` models.

**Tech Used:** `st.session_state` for state management, `st.file_uploader`, `threading` control logic.

<img width="2826" height="1388" alt="image" src="https://github.com/user-attachments/assets/2b1ed01a-0573-4a7e-ab25-956c7477675d" />
<img width="2834" height="1441" alt="image" src="https://github.com/user-attachments/assets/72a33f51-a4e2-4667-bab3-5e6a05031c1a" />
<img width="2821" height="1439" alt="image" src="https://github.com/user-attachments/assets/dde2917c-a133-41c6-ad2d-14402b7a8e15" />
<img width="2819" height="1477" alt="image" src="https://github.com/user-attachments/assets/edd977c2-18eb-48c6-8cf7-7275321f9e12" />
<img width="2787" height="1439" alt="image" src="https://github.com/user-attachments/assets/6b21e90f-54b7-4628-afaf-c9ab8e45fe2d" />


### 2. 📹 Live Streams (Surveillance Dashboard)
**Functionality:**
Real-time visualization of all active sensors.
- **View Modes:** 2x2 Grid, Single Camera Focus, or Split (Day/Thermal).
- **Overlays:** Bounding boxes, confidence scores, and object IDs (DeepSORT).
- **Enhancement:** Real-time visibility improvement for fog/smoke.

**Tech Used:** `OpenCV (cv2)` for frame manipulation, `PIL` for image rendering, `Queue` for threaded video buffering to ensure non-blocking UI.

<img width="2781" height="1409" alt="image" src="https://github.com/user-attachments/assets/ab9c6626-c584-4568-8c41-4553f4d6181e" />
<img width="2834" height="1447" alt="image" src="https://github.com/user-attachments/assets/73912467-17f9-4cf6-ac33-fa5bf3584c01" />
<img width="2837" height="1448" alt="image" src="https://github.com/user-attachments/assets/24c23b14-9199-424d-9654-0ec3154fd9ac" />
<img width="2832" height="1435" alt="image" src="https://github.com/user-attachments/assets/8e1edfcf-ae41-4458-b4ee-b74f3865e9e5" />
<img width="2768" height="1452" alt="image" src="https://github.com/user-attachments/assets/c4072176-47a6-4f35-b09b-3b07009b15d5" />
<img width="2777" height="1487" alt="image" src="https://github.com/user-attachments/assets/269942ad-00c6-485b-b44c-6894811efdd2" />
<img width="2815" height="1452" alt="image" src="https://github.com/user-attachments/assets/7a88ca62-1a76-49bc-8a0d-0bc9332381a3" />


### 3. 📊 Analytics Dashboard
**Functionality:**
A comprehensive data suite providing operational insights.
- **Real-time Metrics:** FPS, System Latency, CPU/GPU Usage.
- **Detection Trends:** Time-series charts showing detection frequency over 1h/6h/24h.
- **Class Breakdown:** Pie charts showing distribution of detected objects (Person vs Vehicle vs Weapon).

**Tech Used:** `Pandas` for data aggregation, `Streamlit Native Charts` (Altair) for interactive visualization, `Psutil` for hardware monitoring.

<img width="2817" height="1366" alt="image" src="https://github.com/user-attachments/assets/1cfc2959-dc34-4ae0-90c6-8946e0cc5c90" />
<img width="2789" height="1443" alt="image" src="https://github.com/user-attachments/assets/58b62bbc-c647-4494-97b0-02780727212b" />
<img width="2813" height="1425" alt="image" src="https://github.com/user-attachments/assets/6e9daa1c-9cff-40eb-b4e8-5ee98b2b75b5" />
<img width="2793" height="1430" alt="image" src="https://github.com/user-attachments/assets/69f98e05-9b26-4f00-b331-312646a63da7" />


### 4. ⚙️ Advanced Model Settings
**Functionality:**
Fine-tune the AI "Brain" without restarting the system.
- **Confidence Threshold:** Slider to filter weak detections (0.0 - 1.0).
- **NMS Threshold:** Adjustment for Non-Maximum Suppression to remove duplicate boxes.
- **Tracking Parameters:** Max lost frames and IOU thresholds for DeepSORT.
<img width="2818" height="1478" alt="image" src="https://github.com/user-attachments/assets/7858de3e-b11f-4b3a-b3c4-62d5e910c654" />
<img width="2805" height="1436" alt="image" src="https://github.com/user-attachments/assets/5c5ff112-5e5f-47bf-b15d-293081f9fec7" />
<img width="2829" height="1413" alt="image" src="https://github.com/user-attachments/assets/99edd11c-1719-4d36-a6a4-c008f58de174" />
<img width="2828" height="1442" alt="image" src="https://github.com/user-attachments/assets/dfa02903-9548-446e-a345-741ca7959864" />
<img width="2839" height="1399" alt="image" src="https://github.com/user-attachments/assets/753e5dc8-16bf-4525-b0b4-f96f65d80677" />
<img width="2795" height="1441" alt="image" src="https://github.com/user-attachments/assets/8978998b-97aa-4631-b7ff-cc60eabafb07" />
<img width="2879" height="1445" alt="image" src="https://github.com/user-attachments/assets/b5a9449b-a6b0-4b3a-8bf4-2eb5769c93fb" />

**Tech Used:** Dynamic parameter injection into running inference threads.

### 5. 🏗️ Architecture & Tech Stack
**Functionality:**
Transparent documentation for engineers.
- **Diagrams:** Interactive Mermaid.js/Graphviz flowcharts showing data pipeline.
- **Dependency Checker:** Live status of installed libraries (PyTorch/CUDA versions).
- **Reasoning:** "Why we chose this stack" comparison tables.

<img width="2792" height="1438" alt="image" src="https://github.com/user-attachments/assets/fe31e322-ef0b-41de-82a1-5e593b8b4209" />
<img width="2547" height="1423" alt="image" src="https://github.com/user-attachments/assets/dc784f74-1d62-4edc-91bc-c69aee2f8a27" />
<img width="2816" height="1470" alt="image" src="https://github.com/user-attachments/assets/ec375d61-5601-4158-b9d3-c18d07043e86" />
<img width="2810" height="1429" alt="image" src="https://github.com/user-attachments/assets/95fbad2b-d401-4b8b-ab4d-315143c24c4d" />

<img width="2806" height="1440" alt="image" src="https://github.com/user-attachments/assets/89d56afd-bd65-4da3-ab3a-b0fb497b5682" />
<img width="2794" height="1445" alt="image" src="https://github.com/user-attachments/assets/39527480-322a-48b9-b32b-2b76f4aeb572" />
<img width="2819" height="1386" alt="image" src="https://github.com/user-attachments/assets/b602a52f-2bac-409f-b1ca-0f9f01863428" />
<img width="2798" height="1467" alt="image" src="https://github.com/user-attachments/assets/28ec8730-fcf0-4bf0-877e-00e9d3e45fe1" />
<img width="2799" height="1387" alt="image" src="https://github.com/user-attachments/assets/6488cad8-b3c9-4f15-824b-e33e46bb17cb" />
<img width="2805" height="1429" alt="image" src="https://github.com/user-attachments/assets/0a3be14b-7602-4beb-8a08-6647e2856659" />
<img width="2804" height="1483" alt="image" src="https://github.com/user-attachments/assets/3d6592e3-6f5e-42d5-af77-b06100a1b4b6" />

**Tech Used:** `Graphviz` for diagrams, `json` for stack exports.

<img width="2799" height="1454" alt="image" src="https://github.com/user-attachments/assets/aec8c771-b56a-46a7-a966-258e22955b77" />
<img width="2759" height="1420" alt="image" src="https://github.com/user-attachments/assets/eac4f62b-cf62-4aa2-92ae-3630cca99316" />
<img width="2609" height="1458" alt="image" src="https://github.com/user-attachments/assets/8e7d055b-9290-447a-8c5a-cc9bdd64e048" />
<img width="2789" height="1426" alt="image" src="https://github.com/user-attachments/assets/e039b67f-304c-411f-91e7-7c8013c5c88b" />
<img width="2796" height="1472" alt="image" src="https://github.com/user-attachments/assets/e02c0a0c-4f28-4d31-b44f-b9aaba4ce7a6" />
<img width="2782" height="1359" alt="image" src="https://github.com/user-attachments/assets/733de45e-6df6-4ca2-9eae-85d47d76dcc3" />
<img width="2769" height="1474" alt="image" src="https://github.com/user-attachments/assets/226badea-d688-4cb3-89b4-77e85dbab3ab" />
<img width="2778" height="1284" alt="image" src="https://github.com/user-attachments/assets/f2e7f1fe-e950-4abf-adc9-d6fd16559f8b" />
<img width="2342" height="1375" alt="image" src="https://github.com/user-attachments/assets/ec0099d9-ea0d-4a9c-a583-c12c24d26b4e" />
<img width="2786" height="1429" alt="image" src="https://github.com/user-attachments/assets/f52171ac-959a-45f9-b761-17858db413cd" />

### 6. 📝 System Logs & Export
**Functionality:**
A robust audit trail for mission debriefing.
- **Live Console:** Auto-scrolling terminal output of all system events.
- **Filtering:** Show only Errors, Warnings, or Info.
- **Export:** Download logs as JSON, CSV, or TXT for external analysis.

**Tech Used:** Custom logging handler, `Pandas` for CSV export, `st.text_area` for console view.

<img width="2809" height="1452" alt="image" src="https://github.com/user-attachments/assets/312d4e2c-7b05-40e9-bfc8-a9666fb3aa02" />
<img width="2786" height="1493" alt="image" src="https://github.com/user-attachments/assets/7f4629f4-2f90-41c0-85c8-bb0e81e039ab" />
<img width="2814" height="1440" alt="image" src="https://github.com/user-attachments/assets/c1e6bef4-adfc-43d7-b83d-c53eba1c7ab8" />

---

## 🏗️ System Architecture

The system follows a modular, pipeline-based architecture designed for scalability and fault tolerance:

```mermaid
graph TD
    A[Sensors: Day/Thermal Cams] -->|RTSP/GigE| B(Pre-processing)
    B -->|Enhancement| C{Drishyak Engine}
    C -->|Enhanced Frames| D[Inference Engine: YOLOv8]
    D -->|Detections| E[Tracker: DeepSORT]
    E -->|Tracks| F[Post-Processing]
    F -->|Data| G[Streamlit Dashboard]
    F -->|Logs| H[Storage / Analytics]
```

---

## 🛠️ Tech Stack

| Category | Technologies / Libraries |
| :--- | :--- |
| **Core Language** | ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) |
| **User Interface** | **Streamlit** (Real-time dashboard), **CSS** (Custom Glassmorphism) |
| **Object Detection** | **YOLOv8** (Ultralytics), **PyTorch** |
| **Inference Engine** | **TensorRT** (FP16 Optimization), **CUDA**, **ONNX Runtime** |
| **Tracking Algorithm**| **DeepSORT** (Re-ID), **Kalman Filter**, **Hungarian Algorithm** |
| **Image Processing** | **OpenCV** (cv2), **Pillow**, **NumPy**, **SciPy** |
| **Enhancement**| **CLAHE** (Contrast Limited AHE), **Dark Channel Prior** (Dehazing) |
| **Hardware Support** | **NVIDIA Jetson Orin AGX**, GigE Machine Vision Cameras |
| **Data & Logs** | **Pandas** (Analytics), **JSON** (Config), **Logging** (System Events) |
| **Deployment** | **Docker**, **NVIDIA JetPack 5.1+** |

---

## 🔧 Hardware Requirements

### Minimum (Development)
- **CPU**: Intel i5 / AMD Ryzen 5 (8th gen+)
- **RAM**: 16 GB
- **GPU**: NVIDIA GTX 1060 (6GB VRAM) or better
- **Storage**: 50 GB SSD
- **OS**: Windows 10+ / Ubuntu 20.04+

### Production (Deployment)
- **Platform**: NVIDIA Jetson Orin AGX (64GB)
- **Cameras**: 4× GigE cameras (2× Day, 2× Thermal LWIR)
- **Storage**: 1 TB NVMe SSD (Industrial Grade)
- **Power**: 24V DC, rugged enclosure

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/Ratnesh-181998/Defence-AI-Multisensor-Surveillance-YOLOv8.git
cd Defence-AI-Multisensor-Surveillance-YOLOv8
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# For Standard Usage (Streamlit Cloud / CPU)
pip install -r requirements.txt

# For GPU/Jetson Development (Uncomment specific lines in requirements.txt first)
# pip install -r requirements.txt 
```

---

## 🚀 Quick Start

Run the main application:
```bash
streamlit run app.py
```
*The application will launch in your default web browser at `http://localhost:8501`*

---

## 📖 Usage Guide

1.  **Control Panel**: Select active camera inputs (Day-1, Thermal-1, etc.) and click "▶️ ENGAGE" to start the system.
2.  **Live Streams**: Monitor real-time feeds with detection overlays. Use the "Snapshot" button to capture evidence.
3.  **Analytics**: View statistical breakdowns of detections over time.
4.  **Model Settings**: Fine-tune confidence thresholds, IoU, and tracking parameters dynamically.
5.  **Logs**: Review, filter, and export system event logs.

---

## 🚢 Deployment

### Streamlit Cloud
1.  Fork this repository.
2.  Login to [Streamlit Cloud](https://streamlit.io/cloud).
3.  Create a new app pointing to your forked repo.
4.  Select `app.py` as the main file.
5.  **Note**: Ensure `requirements.txt` is optimized for headless environments (opencv-headless).

### Jetson Orin (Docker)
```bash
# Build Docker image
docker build -t defence-ai:jetson .

# Run container with GPU access
docker run --runtime nvidia --network host --privileged defence-ai:jetson
```

---

## 📊 Performance Metrics

| Component | Latency (ms) | FPS |
|-----------|--------------|-----|
| Camera Capture | 40 | 25 |
| Preprocessing | 15 | - |
| YOLOv8m Inference | 280 | 3.6 |
| Tracking (DeepSORT) | 25 | - |
| **Total Pipeline** | **~400** | **2.5** |

*Benchmarks recorded on NVIDIA Jetson Orin AGX 64GB in Max Power mode.*

---

## 📞 Contact

**RATNESH SINGH**

- 📧 **Email**: [rattudacsit2021gate@gmail.com](mailto:rattudacsit2021gate@gmail.com)
- 💼 **LinkedIn**: [ratneshkumar1998](https://www.linkedin.com/in/ratneshkumar1998/)
- 🐙 **GitHub**: [Ratnesh-181998](https://github.com/Ratnesh-181998)
- 📱 **Phone**: +91-947XXXXX46

### Project Links
- 🌐 **Live Demo**: [Streamlit App](https://defence-ai-multisensor-surveillance-yolov8-vusybzt9bohpykhzkq3.streamlit.app/)
- 📖 **Documentation**: [GitHub Wiki](https://github.com/Ratnesh-181998/Defence-AI-Multisensor-Surveillance-YOLOv8/wiki)
- 🐛 **Issue Tracker**: [GitHub Issues](https://github.com/Ratnesh-181998/Defence-AI-Multisensor-Surveillance-YOLOv8/issues)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<div align="center">
  <b>⭐ Star this repo if you find it useful! ⭐</b><br>
  Made with ❤️ by Ratnesh Singh
</div>
