"""
Dataset Preparation Script
Organizes and validates your track error dataset
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import yaml
import json
from typing import Dict, List, Tuple

class DatasetOrganizer:
    def __init__(self, config_path='config.yaml'):
        """Initialize dataset organizer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_root = Path('data')
        self.raw_dir = self.data_root / 'raw'
        self.dataset_dir = self.data_root / 'dataset'
        
        # Create severity subdirectories
        self.severity_dirs = ['simple', 'moderate', 'severe']
        self.split_dirs = ['train', 'valid', 'test']
        
    def organize_dataset(self, split_ratio=(0.7, 0.2, 0.1)):
        """
        Organize dataset into train/val/test splits with severity subdirectories.
        
        Args:
            split_ratio: Tuple of (train, val, test) ratios
        """
        print("Organizing dataset...")
        
        # Create directory structure
        for split in self.split_dirs:
            for severity in self.severity_dirs:
                (self.dataset_dir / split / severity).mkdir(parents=True, exist_ok=True)
        
        # Get all images from raw directory
        image_files = list(self.raw_dir.glob('**/*.jpg')) + \
                     list(self.raw_dir.glob('**/*.jpeg')) + \
                     list(self.raw_dir.glob('**/*.png'))
        
        if not image_files:
            print("[ERROR] No images found in data/raw/")
            print("[INFO] Please add your track error images to data/raw/")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Organize images
        for img_path in image_files:
            severity = self._get_severity_from_path(img_path)
            split = self._assign_split(split_ratio)
            
            dest = self.dataset_dir / split / severity / img_path.name
            shutil.copy2(img_path, dest)
        
        print("[OK] Dataset organized successfully!")
        self.print_statistics()
    
    def _get_severity_from_path(self, path: Path) -> str:
        """Extract severity from path (folder name or filename)."""
        path_str = str(path).lower()
        
        # Check parent folder
        parent = path.parent.name.lower()
        for severity in self.severity_dirs:
            if severity in parent:
                return severity
        
        # Check filename
        for severity in self.severity_dirs:
            if severity in path.name.lower():
                return severity
        
        return 'simple'  # Default
    
    def _assign_split(self, split_ratio: Tuple[float, float, float]) -> str:
        """Randomly assign split based on ratio."""
        import random
        rand = random.random()
        
        if rand < split_ratio[0]:
            return 'train'
        elif rand < split_ratio[0] + split_ratio[1]:
            return 'valid'
        else:
            return 'test'
    
    def print_statistics(self):
        """Print dataset statistics."""
        stats = defaultdict(lambda: defaultdict(int))
        
        for split in self.split_dirs:
            for severity in self.severity_dirs:
                split_dir = self.dataset_dir / split / severity
                count = len(list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png')))
                stats[split][severity] = count
        
        print("\n[STATS] Dataset Statistics:")
        print("-" * 50)
        for split in self.split_dirs:
            total = sum(stats[split].values())
            print(f"\n{split.upper()}: {total} images")
            for severity in self.severity_dirs:
                count = stats[split][severity]
                if count > 0:
                    print(f"  {severity}: {count}")

    def validate_dataset(self):
        """Validate dataset structure."""
        print("Validating dataset...")
        issues = []
        
        for split in self.split_dirs:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                issues.append(f"Missing directory: {split_dir}")
                continue
            
            for severity in self.severity_dirs:
                severity_dir = split_dir / severity
                if not severity_dir.exists():
                    issues.append(f"Empty directory: {severity_dir}")
        
        if issues:
            print("[ERROR] Validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("[OK] Dataset validation passed!")
            return True

def main():
    organizer = DatasetOrganizer()
    
    # Validate first
    if organizer.validate_dataset():
        print("\n[OK] Dataset is ready!")
    else:
        print("\n[INFO] Run organize_dataset() to organize your data")
        print("[TIP] Place your images in data/raw/ with severity in folder names")

if __name__ == "__main__":
    main()

