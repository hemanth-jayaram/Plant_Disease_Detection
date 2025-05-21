import kagglehub
import os
import shutil
from pathlib import Path

def download_and_prepare_dataset():
    # Download the dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download("rashikrahmanpritom/plant-disease-recognition-dataset")
    print(f"Dataset downloaded to: {path}")
    
    # Create data directory structure
    data_dir = Path('data')
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get the path to the actual images
    train_path = Path(path) / 'Train' / 'Train'
    val_path = Path(path) / 'Validation' / 'Validation'
    
    # Copy training data
    print("Copying training data...")
    for class_dir in os.listdir(train_path):
        class_path = train_path / class_dir
        if not class_path.is_dir():
            continue
            
        # Create corresponding directory in train
        os.makedirs(train_dir / class_dir, exist_ok=True)
        
        # Copy all images to train directory
        for img in class_path.glob('*'):
            try:
                shutil.copy2(img, train_dir / class_dir / img.name)
            except PermissionError:
                print(f"Permission error copying {img}, skipping...")
    
    # Copy validation data
    print("Copying validation data...")
    for class_dir in os.listdir(val_path):
        class_path = val_path / class_dir
        if not class_path.is_dir():
            continue
            
        # Create corresponding directory in val
        os.makedirs(val_dir / class_dir, exist_ok=True)
        
        # Copy all images to validation directory
        for img in class_path.glob('*'):
            try:
                shutil.copy2(img, val_dir / class_dir / img.name)
            except PermissionError:
                print(f"Permission error copying {img}, skipping...")
    
    print("Dataset preparation complete!")
    print(f"Number of classes: {len(os.listdir(train_dir))}")
    print(f"Total training images: {sum(len(files) for _, _, files in os.walk(train_dir))}")
    print(f"Total validation images: {sum(len(files) for _, _, files in os.walk(val_dir))}")

if __name__ == '__main__':
    download_and_prepare_dataset()
