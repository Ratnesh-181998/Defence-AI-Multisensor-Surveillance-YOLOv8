import yaml
from pathlib import Path
import shutil
import os

def create_sample_dataset(root_dir: str = "data/sample_dataset"):
    """
    Creates a valid YOLOv8 dataset directory structure with a sample data.yaml
    """
    root = Path(root_dir)
    images_train = root / "images" / "train"
    images_val = root / "images" / "val"
    labels_train = root / "labels" / "train"
    labels_val = root / "labels" / "val"
    
    # Create directories
    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)
        
    print(f"Created dataset structure at: {root}")
    
    # Create data.yaml content
    data_yaml = {
        'path': str(root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'person',
            1: 'vehicle',
            2: 'weapon'
        }
    }
    
    yaml_path = root / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
        
    print(f"Created data.yaml at: {yaml_path}")
    print("\n[NOTE] To proceed, please place your images in images/train and labels in labels/train")
    return yaml_path

if __name__ == "__main__":
    create_sample_dataset()
