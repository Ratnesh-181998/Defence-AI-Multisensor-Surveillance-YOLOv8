"""
Generate a sample dataset for training YOLOv8 models.
Creates synthetic images with bounding boxes for demonstration purposes.
"""

import os
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

def create_sample_image_with_annotations(img_path, label_path, img_size=(640, 640)):
    """
    Create a synthetic image with random objects and corresponding YOLO annotations.
    
    Args:
        img_path: Path to save the image
        label_path: Path to save the YOLO format label
        img_size: Image dimensions (width, height)
    """
    # Create a random background
    img = Image.new('RGB', img_size, color=(
        random.randint(50, 150),
        random.randint(50, 150),
        random.randint(50, 150)
    ))
    draw = ImageDraw.Draw(img)
    
    # Define classes
    classes = ['person', 'vehicle', 'weapon']
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
    
    annotations = []
    num_objects = random.randint(1, 5)  # 1-5 objects per image
    
    for _ in range(num_objects):
        # Random class
        class_id = random.randint(0, len(classes) - 1)
        
        # Random bounding box (ensure it's within image bounds)
        width = random.randint(50, 200)
        height = random.randint(50, 200)
        x1 = random.randint(0, img_size[0] - width)
        y1 = random.randint(0, img_size[1] - height)
        x2 = x1 + width
        y2 = y1 + height
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=colors[class_id], width=3)
        
        # Add label text
        try:
            draw.text((x1, y1 - 15), classes[class_id], fill=colors[class_id])
        except:
            pass  # Skip if font not available
        
        # Convert to YOLO format (normalized)
        # YOLO format: class_id center_x center_y width height (all normalized 0-1)
        center_x = ((x1 + x2) / 2) / img_size[0]
        center_y = ((y1 + y2) / 2) / img_size[1]
        norm_width = width / img_size[0]
        norm_height = height / img_size[1]
        
        annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
    
    # Save image
    img.save(img_path)
    
    # Save annotations
    with open(label_path, 'w') as f:
        f.write('\n'.join(annotations))

def generate_dataset(output_dir='data', num_train=100, num_val=20):
    """
    Generate a complete YOLO dataset with train and validation splits.
    
    Args:
        output_dir: Root directory for the dataset
        num_train: Number of training images
        num_val: Number of validation images
    """
    output_path = Path(output_dir)
    
    # Create directory structure
    dirs = [
        output_path / 'images' / 'train',
        output_path / 'images' / 'val',
        output_path / 'labels' / 'train',
        output_path / 'labels' / 'val'
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("📁 Created directory structure")
    
    # Generate training images
    print(f"🎨 Generating {num_train} training images...")
    for i in range(num_train):
        img_path = output_path / 'images' / 'train' / f'train_{i:04d}.jpg'
        label_path = output_path / 'labels' / 'train' / f'train_{i:04d}.txt'
        create_sample_image_with_annotations(img_path, label_path)
        
        if (i + 1) % 20 == 0:
            print(f"  ✓ Generated {i + 1}/{num_train} training images")
    
    # Generate validation images
    print(f"🎨 Generating {num_val} validation images...")
    for i in range(num_val):
        img_path = output_path / 'images' / 'val' / f'val_{i:04d}.jpg'
        label_path = output_path / 'labels' / 'val' / f'val_{i:04d}.txt'
        create_sample_image_with_annotations(img_path, label_path)
    
    print(f"  ✓ Generated {num_val} validation images")
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'person',
            1: 'vehicle',
            2: 'weapon'
        },
        'nc': 3  # number of classes
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml at {yaml_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Dataset Generation Complete!")
    print("="*60)
    print(f"📁 Location: {output_path.absolute()}")
    print(f"🎯 Classes: person, vehicle, weapon")
    print(f"📸 Training images: {num_train}")
    print(f"📸 Validation images: {num_val}")
    print(f"📄 Config file: {yaml_path}")
    print("\n💡 Next steps:")
    print("   1. Review the generated images in data/images/")
    print("   2. Go to Control Panel → AI Core Config")
    print("   3. Click 'Train Model' button")
    print("   4. Configure training parameters")
    print("   5. Click 'Start Training Now'")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample YOLO dataset")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--train", type=int, default=100, help="Number of training images")
    parser.add_argument("--val", type=int, default=20, help="Number of validation images")
    
    args = parser.parse_args()
    
    generate_dataset(args.output, args.train, args.val)
