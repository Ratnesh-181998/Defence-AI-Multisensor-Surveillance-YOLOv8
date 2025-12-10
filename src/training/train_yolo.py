import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_yolo(config_path: str, data_yaml: str, output_dir: str):
    """
    Execute YOLOv8 training pipeline.
    
    Args:
        config_path: Path to hyperparameter config (training.yaml)
        data_yaml: Path to dataset definition (data.yaml)
        output_dir: Directory to save results
    """
    
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    logging.info(f"Loaded configuration from {config_path}")
    logging.info(f"Model: {cfg['model']['name']}, Epochs: {cfg['train']['epochs']}")

    # 2. Initialize Model
    try:
        model_name = f"{cfg['model']['name']}.pt"
        model = YOLO(model_name)
        logging.info(f"Initialized {model_name} (Pretrained: {cfg['model']['pretrained']})")
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        return

    # 3. Configure Training Arguments
    project_dir = Path(output_dir)
    name_dir = "exp_" + cfg['model']['name']
    
    # Map config to YOLO args
    args = {
        'data': data_yaml,
        'epochs': cfg['train']['epochs'],
        'batch': cfg['train']['batch_size'],
        'imgsz': cfg['model']['input_size'],
        'device': cfg['device'],
        'workers': cfg['workers'],
        'project': str(project_dir),
        'name': name_dir,
        'exist_ok': True,
        'pretrained': cfg['model']['pretrained'],
        'optimizer': cfg['train']['optimizer'],
        'lr0': cfg['train']['lr0'],
        'patience': cfg['train']['patience'],
        
        # Augmentation overrides
        'hsv_h': cfg['augmentation']['hsv_h'],
        'hsv_s': cfg['augmentation']['hsv_s'],
        'hsv_v': cfg['augmentation']['hsv_v'],
        'mosaic': cfg['augmentation']['mosaic'],
    }

    # 4. Start Training
    logging.info("Starting training run...")
    try:
        results = model.train(**args)
        logging.info(f"Training completed successfully. Results saved to {project_dir / name_dir}")
        
        # Example export to ONNX after training
        success = model.export(format='onnx')
        if success:
            logging.info(f"Model exported to ONNX: {success}")
            
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defence AI - YOLOv8 Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--output", type=str, default="models/trained", help="Output directory")
    
    args = parser.parse_args()
    
    train_yolo(args.config, args.data, args.output)
