"""
YOLOv8 Training Script for Rack and Tile Detection

This script trains a YOLOv8 model to detect:
- Class 0: rack (the tile holder)
- Class 1: tile (individual scrabble tiles)

Usage:
    python training/train_rack_tile.py
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import yaml


def setup_dataset(source_images_dir: str, source_labels_dir: str, 
                  output_dir: str, val_split: float = 0.2) -> str:
    """
    Organize the dataset into YOLO-expected structure with train/val splits.
    
    Args:
        source_images_dir: Path to source images
        source_labels_dir: Path to source YOLO annotations
        output_dir: Output directory for organized dataset
        val_split: Fraction of data to use for validation
    
    Returns:
        Path to the data.yaml config file
    """
    output_path = Path(output_dir)
    
    # Create directory structure
    dirs = [
        output_path / "images" / "train",
        output_path / "images" / "val",
        output_path / "labels" / "train",
        output_path / "labels" / "val",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get list of images with corresponding labels
    source_images = Path(source_images_dir)
    source_labels = Path(source_labels_dir)
    
    image_files = list(source_images.glob("*.jpg")) + list(source_images.glob("*.png"))
    
    # Filter to only include images that have corresponding label files
    valid_pairs = []
    for img_path in image_files:
        label_path = source_labels / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
    
    print(f"Found {len(valid_pairs)} valid image-label pairs")
    
    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(valid_pairs)
    
    split_idx = int(len(valid_pairs) * (1 - val_split))
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"Train set: {len(train_pairs)} images")
    print(f"Val set: {len(val_pairs)} images")
    
    # Copy files to respective directories
    def copy_pairs(pairs, split_name):
        for img_path, label_path in pairs:
            # Copy image
            shutil.copy2(img_path, output_path / "images" / split_name / img_path.name)
            # Copy label
            shutil.copy2(label_path, output_path / "labels" / split_name / label_path.name)
    
    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    
    # Create data.yaml config
    data_config = {
        "path": str(output_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "rack",
            1: "tile"
        }
    }
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Dataset config saved to: {yaml_path}")
    return str(yaml_path)


def train_yolo(data_yaml: str, epochs: int = 100, imgsz: int = 640, 
               model_size: str = "n", project_dir: str = "training/runs"):
    """
    Train YOLOv8 model on the prepared dataset.
    
    Args:
        data_yaml: Path to data.yaml config file
        epochs: Number of training epochs
        imgsz: Input image size
        model_size: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        project_dir: Directory to save training runs
    
    Returns:
        Path to the best trained model
    """
    # Load a pretrained YOLOv8 model
    model_name = f"yolov8{model_size}.pt"
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Train the model
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Image size: {imgsz}")
    print(f"Project directory: {project_dir}")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project_dir,
        name="rack_tile_detector",
        patience=20,  # Early stopping patience
        save=True,
        plots=True,
        device="mps" if os.uname().sysname == "Darwin" else "0",  # Use MPS on Mac, GPU 0 otherwise
    )
    
    # Get path to best model
    best_model_path = Path(project_dir) / "rack_tile_detector" / "weights" / "best.pt"
    
    if best_model_path.exists():
        # Copy best model to training directory with a cleaner name
        final_model_path = Path("training") / "rack_tile_yolov8.pt"
        shutil.copy2(best_model_path, final_model_path)
        print(f"\nBest model saved to: {final_model_path}")
        return str(final_model_path)
    
    return str(best_model_path)


def main():
    # Paths
    source_images = "data/rack_tile/images"
    source_labels = "data/rack_tile/yolo_annotations"
    dataset_output = "data/rack_tile/yolo_dataset"
    
    # Check if source directories exist
    if not os.path.exists(source_images):
        raise FileNotFoundError(f"Images directory not found: {source_images}")
    if not os.path.exists(source_labels):
        raise FileNotFoundError(f"Labels directory not found: {source_labels}")
    
    print("=" * 60)
    print("YOLOv8 Rack & Tile Detection Training")
    print("=" * 60)
    
    # Step 1: Setup dataset
    print("\n[Step 1] Setting up dataset...")
    data_yaml = setup_dataset(
        source_images_dir=source_images,
        source_labels_dir=source_labels,
        output_dir=dataset_output,
        val_split=0.2
    )
    
    # Step 2: Train model
    print("\n[Step 2] Training YOLOv8 model...")
    model_path = train_yolo(
        data_yaml=data_yaml,
        epochs=100,
        imgsz=640,
        model_size="n",  # Using nano for faster training, can change to "s" or "m" for better accuracy
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)
    
    # Step 3: Quick validation
    print("\n[Step 3] Running validation on trained model...")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
