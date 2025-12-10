"""
Quick training script for the Defence AI project.
Trains a YOLOv8 model on the generated sample dataset.
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

def quick_train():
    """
    Quick training function with sensible defaults for demo purposes.
    """
    print("="*60)
    print("🚀 Defence AI - Quick Model Training")
    print("="*60)
    
    # Check if data.yaml exists
    data_yaml = Path("data/data.yaml")
    if not data_yaml.exists():
        print("❌ Error: data/data.yaml not found!")
        print("💡 Run: python scripts/generate_sample_dataset.py first")
        return
    
    print(f"✅ Found dataset config: {data_yaml}")
    
    # Load data config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"📊 Dataset: {data_config['nc']} classes")
    print(f"🎯 Classes: {', '.join(data_config['names'].values())}")
    
    # Initialize model (using nano for quick training)
    print("\n🧠 Initializing YOLOv8 Nano model...")
    model = YOLO('yolov8n.pt')  # Nano model - fastest for demo
    
    # Training parameters (optimized for quick demo)
    print("\n⚙️ Training Configuration:")
    train_params = {
        'data': str(data_yaml),
        'epochs': 10,  # Quick demo - use 100+ for real training
        'batch': 8,    # Small batch for compatibility
        'imgsz': 640,
        'device': 'cpu',  # Use CPU for compatibility (change to '0' for GPU)
        'project': 'models/trained',
        'name': 'defence_ai_demo',
        'exist_ok': True,
        'patience': 5,
        'save': True,
        'plots': True,
        'verbose': True
    }
    
    for key, value in train_params.items():
        print(f"  • {key}: {value}")
    
    # Start training
    print("\n🎓 Starting training...")
    print("⏱️  This will take a few minutes...")
    print("-"*60)
    
    try:
        results = model.train(**train_params)
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        print(f"📁 Model saved to: models/trained/defence_ai_demo/weights/best.pt")
        print(f"📊 Results saved to: models/trained/defence_ai_demo/")
        
        # Export to ONNX
        print("\n📦 Exporting to ONNX format...")
        try:
            onnx_path = model.export(format='onnx')
            print(f"✅ ONNX model: {onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")
        
        print("\n💡 Next steps:")
        print("   1. Check models/trained/defence_ai_demo/ for results")
        print("   2. View training plots in the results folder")
        print("   3. Use best.pt for inference in the app")
        print("   4. Upload best.pt in Control Panel → Load Model Weights")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Ensure ultralytics is installed: pip install ultralytics")
        print("   • Check if dataset images exist in data/images/")
        print("   • Try reducing batch size or epochs")
        raise

if __name__ == "__main__":
    quick_train()
