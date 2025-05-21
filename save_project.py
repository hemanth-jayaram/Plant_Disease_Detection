import os
import shutil
from pathlib import Path

def save_project():
    # Define source and destination directories
    src_dir = Path(__file__).parent
    dest_dir = src_dir / 'saved_project'
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(exist_ok=True)
    
    # List of files to save
    files_to_save = [
        'src/train.py',
        'src/predict.py',
        'requirements.txt',
        'README.md',
        'models/best_model.pth'
    ]
    
    # Copy each file
    for file_path in files_to_save:
        src_file = src_dir / file_path
        if src_file.exists():
            # Create necessary subdirectories in destination
            dest_file = dest_dir / file_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(src_file, dest_file)
            print(f'Copied: {file_path}')
        else:
            print(f'Skipped (not found): {file_path}')

if __name__ == '__main__':
    save_project()
