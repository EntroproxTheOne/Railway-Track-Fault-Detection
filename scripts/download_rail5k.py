"""
Download and prepare Rail-5k dataset for track error detection
Dataset: https://github.com/TommyZihao/Rail-5k-dataset
"""

import os
import subprocess
import shutil
from pathlib import Path
import requests
import zipfile

def clone_repository():
    """Clone the Rail-5k dataset repository."""
    repo_url = "https://github.com/TommyZihao/Rail-5k-dataset.git"
    target_dir = Path("data/raw/rail-5k-temp")
    
    print("[INFO] Cloning Rail-5k dataset repository...")
    print(f"[URL] {repo_url}")
    
    # Create target directory
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Clone repository using git
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(target_dir)],
            check=True,
            capture_output=True,
            text=True
        )
        print("[OK] Repository cloned successfully!")
        return target_dir
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to clone repository: {e}")
        print("[INFO] Make sure git is installed: https://git-scm.com/")
        return None
    except FileNotFoundError:
        print("[ERROR] Git not found!")
        print("[INFO] Install git from: https://git-scm.com/")
        print("[INFO] Or manually download the dataset from GitHub")
        return None

def download_manual():
    """Instructions for manual download."""
    print("\n[MANUAL DOWNLOAD INSTRUCTIONS]")
    print("=" * 60)
    print("If git clone doesn't work, follow these steps:")
    print()
    print("1. Visit: https://github.com/TommyZihao/Rail-5k-dataset")
    print("2. Click 'Code' button → 'Download ZIP'")
    print("3. Extract the ZIP file")
    print("4. Copy contents to: data/raw/rail-5k-temp/")
    print()
    print("Then run: python scripts/download_rail5k.py --prepare")
    print("=" * 60)

def prepare_dataset(temp_dir):
    """Organize Rail-5k dataset for our training structure."""
    if temp_dir is None or not temp_dir.exists():
        print("[ERROR] Dataset directory not found!")
        return False
    
    print(f"[INFO] Preparing dataset from: {temp_dir}")
    
    # Define where to put organized data
    output_dir = Path("data/raw")
    
    # Check what's in the dataset
    print("\n[INFO] Scanning dataset structure...")
    
    # Look for common dataset structures
    image_dir = None
    label_dir = None
    
    # Common locations
    possible_image_dirs = [
        temp_dir / "images",
        temp_dir / "train" / "images",
        temp_dir / "data" / "images",
        temp_dir / "dataset" / "images",
    ]
    
    possible_label_dirs = [
        temp_dir / "labels",
        temp_dir / "annotations",
        temp_dir / "train" / "labels",
        temp_dir / "data" / "labels",
        temp_dir / "dataset" / "labels",
    ]
    
    # Find images
    for img_dir in possible_image_dirs:
        if img_dir.exists() and list(img_dir.glob("*.jpg")) or list(img_dir.glob("*.png")):
            image_dir = img_dir
            print(f"[FOUND] Images in: {image_dir}")
            break
    
    # Find labels
    for lbl_dir in possible_label_dirs:
        if lbl_dir.exists() and list(lbl_dir.glob("*.txt")) or list(lbl_dir.glob("*.xml")):
            label_dir = lbl_dir
            print(f"[FOUND] Labels in: {label_dir}")
            break
    
    if not image_dir:
        print("[ERROR] Could not find images in expected locations")
        print("[INFO] Please check the dataset structure")
        print(f"[INFO] Looking in: {temp_dir}")
        return False
    
    # Count images
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    print(f"[INFO] Found {len(images)} images")
    
    if len(images) == 0:
        print("[ERROR] No images found in the dataset!")
        print("[INFO] Please check the dataset structure")
        return False
    
    # Copy images to our structure
    # Since we don't know severity, we'll put them all in a general folder
    # User can organize them later
    print("\n[INFO] Copying images to: data/raw/")
    
    # Create general folder
    general_dir = output_dir / "rail-5k-general"
    general_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy first 100 images as sample (to avoid too many)
    print("[INFO] Copying sample images (first 100)...")
    count = 0
    for img in images[:100]:
        shutil.copy2(img, general_dir / img.name)
        count += 1
        if count % 10 == 0:
            print(f"  Copied {count} images...")
    
    print(f"[OK] Copied {count} sample images to data/raw/rail-5k-general/")
    print(f"[INFO] Total images in dataset: {len(images)}")
    
    # If there are labels, copy them too
    if label_dir:
        print(f"[INFO] Found labels in: {label_dir}")
        label_output = Path("data/annotations/rail-5k")
        label_output.mkdir(parents=True, exist_ok=True)
        
        labels = list(label_dir.glob("*.txt")) + list(label_dir.glob("*.xml"))
        for lbl in labels[:100]:
            shutil.copy2(lbl, label_output / lbl.name)
        
        print(f"[OK] Copied {min(100, len(labels))} annotation files")
    
    print("\n[SUCCESS] Dataset preparation complete!")
    print("\n[NEXT STEPS]")
    print("1. Review images in: data/raw/rail-5k-general/")
    print("2. Organize by severity:")
    print("   - Move simple errors to: data/raw/simple/")
    print("   - Move moderate errors to: data/raw/moderate/")
    print("   - Move severe errors to: data/raw/severe/")
    print("3. Run: python scripts/prepare_dataset.py")
    
    return True

def create_sample_structure():
    """Create a sample dataset structure for testing."""
    print("\n[INFO] Creating sample dataset structure...")
    
    sample_dir = Path("data/raw/simple")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[OK] Created: {sample_dir}")
    print("\n[INFO] The Rail-5k repository contains only code, not images.")
    print("[INFO] To use real data, you have two options:")
    print("\n[OPTION 1] Download from public datasets:")
    print("  - Kaggle: Search 'railway track defect'")
    print("  - Google Dataset Search: railway infrastructure")
    print("  - Visit: https://datasetsearch.research.google.com/")
    print("\n[OPTION 2] Use your own images:")
    print("  - Take photos at railway stations")
    print("  - Add to: data/raw/simple/, data/raw/moderate/, data/raw/severe/")
    print("\n[OPTION 3] Start training with 0 images (YOLO will use transfer learning)")
    print("  - The model will use pre-trained COCO weights")
    print("  - You can fine-tune later when you have data")
    
    return True

def main():
    """Main download and preparation workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Rail-5k Dataset')
    parser.add_argument('--prepare-only', action='store_true',
                       help='Only prepare existing dataset, skip download')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep temporary directory after preparation')
    parser.add_argument('--sample', action='store_true',
                       help='Create sample directory structure only')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  RAIL-5K DATASET DOWNLOADER")
    print("  https://github.com/TommyZihao/Rail-5k-dataset")
    print("=" * 60)
    print()
    
    # If sample flag, just create structure
    if args.sample:
        create_sample_structure()
        return
    
    temp_dir = Path("data/raw/rail-5k-temp")
    
    # Download if not prepare-only
    if not args.prepare_only:
        if temp_dir.exists():
            print("[INFO] Dataset already exists in data/raw/rail-5k-temp/")
            response = input("Re-download? (y/n): ").lower()
            if response == 'y':
                shutil.rmtree(temp_dir)
                temp_dir = clone_repository()
        else:
            temp_dir = clone_repository()
        
        if temp_dir is None:
            download_manual()
            return
    else:
        if not temp_dir.exists():
            print("[ERROR] Dataset not found!")
            print("[INFO] Remove --prepare-only flag to download first")
            return
    
    # Prepare dataset
    if temp_dir.exists():
        success = prepare_dataset(temp_dir)
        
        # Clean up temporary directory
        if success and not args.keep_temp:
            print("\n[CLEANUP] Removing temporary directory...")
            try:
                shutil.rmtree(temp_dir)
                print("[OK] Cleanup complete")
            except Exception as e:
                print(f"[WARNING] Could not remove temp directory: {e}")
    else:
        print("\n[INFO] Creating sample structure for you...")
        create_sample_structure()

if __name__ == "__main__":
    main()

