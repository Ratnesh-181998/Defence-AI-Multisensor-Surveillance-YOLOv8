from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

@dataclass
class CameraConfig:
    id: int
    name: str
    source: Any  # int for device index, str for file/url
    fps: int = 30
    resolution: tuple = (640, 480)

class Config:
    # Project Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    EXAMPLES_DIR = PROJECT_ROOT / "examples"

    # Default Cameras
    DEFAULT_CAMERAS = [
        CameraConfig(0, "Day-1", 0),
        CameraConfig(1, "Day-2", 1),
        CameraConfig(2, "Thermal-1", str(EXAMPLES_DIR / "videos/thermal_sample.mp4")),
        CameraConfig(3, "Thermal-2", str(EXAMPLES_DIR / "videos/thermal_sample.mp4")),
    ]

    # Model Settings
    DEFAULT_MODEL = "yolov8m"
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.45
    
    # System Settings
    ENABLE_CUDA = True
    ENABLE_FP16 = True
    
    @classmethod
    def setup_dirs(cls):
        """Create necessary directories if they don't exist"""
        for p in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR, cls.EXAMPLES_DIR]:
            p.mkdir(parents=True, exist_ok=True)
            
Config.setup_dirs()
