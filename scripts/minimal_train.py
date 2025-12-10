"""
Minimal training script - trains for just 3 epochs for quick demonstration.
"""

from ultralytics import YOLO
import torch

print("="*60)
print("🚀 Defence AI - Minimal Training (3 epochs)")
print("="*60)

# Initialize model
print("🧠 Loading YOLOv8 Nano...")
model = YOLO('yolov8n.pt')

# Minimal training
print("\n🎓 Training for 3 epochs (quick demo)...")
try:
    results = model.train(
        data='data/data.yaml',
        epochs=3,
        batch=4,
        imgsz=320,  # Smaller for speed
        device='cpu',
        project='models/trained',
        name='quick_demo',
        exist_ok=True,
        verbose=False,
        plots=False
    )
    
    print("\n✅ Training Complete!")
    print(f"📁 Model saved: models/trained/quick_demo/weights/best.pt")
    print("\n💡 Upload this model in the app's Control Panel!")
    
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
