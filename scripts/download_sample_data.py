"""
Sample Dataset Download Script
Download sample images from public sources for testing
"""

import urllib.request
import os
from pathlib import Path
import zipfile
from tqdm import tqdm

def download_file(url: str, save_path: str):
    """Download file with progress bar."""
    print(f"📥 Downloading: {url}")
    
    def show_progress(block_num, block_size, total_size):
        pbar.update(block_size)
    
    pbar = tqdm(total=None)
    urllib.request.urlretrieve(url, save_path, reporthook=show_progress)
    pbar.close()
    print(f"✅ Downloaded: {save_path}")

def download_sample_images():
    """Download sample railway track images for testing."""
    
    # Create data directory
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("📦 Sample dataset links:")
    print("=" * 60)
    print("\n🔗 Recommended sources:")
    print("1. Kaggle: https://www.kaggle.com/datasets")
    print("   Search: 'railway track', 'train track', 'railway defect'")
    print("\n2. Google Dataset Search:")
    print("   https://datasetsearch.research.google.com/")
    print("\n3. Roboflow Universe:")
    print("   https://universe.roboflow.com/")
    print("\n4. Papers with Code:")
    print("   https://paperswithcode.com/")
    
    print("\n💡 To download from Kaggle:")
    print("1. Install kaggle: pip install kaggle")
    print("2. Setup credentials: kaggle competitions download -c <dataset>")
    print("3. Extract to data/raw/")
    
    print("\n📝 Alternative: Manual download")
    print("1. Visit dataset sites above")
    print("2. Download images of railway tracks")
    print("3. Organize in data/raw/ with severity folders")
    print("   - data/raw/simple/")
    print("   - data/raw/moderate/")
    print("   - data/raw/severe/")

def create_synthetic_dataset():
    """Create a synthetic dataset using image transformations."""
    print("\n🎨 Creating synthetic dataset from base images...")
    print("💡 This will use Albumentations to augment your images")
    
    # Check if base images exist
    base_dir = Path('data/raw')
    if not list(base_dir.glob('*.jpg')) and not list(base_dir.glob('*.png')):
        print("❌ No base images found in data/raw/")
        print("📝 Please add some sample images first")
        return
    
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from PIL import Image
        import random
        
        # Define augmentation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.3),
        ])
        
        print("🔧 Applying augmentations...")
        
        # Get all images
        images = list(base_dir.glob('*.jpg')) + list(base_dir.glob('*.png'))
        
        for img_path in tqdm(images, desc="Augmenting"):
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            # Apply augmentation 5 times
            for i in range(5):
                aug_img = transform(image=img_array)['image']
                aug_img_pil = Image.fromarray(aug_img)
                
                # Save augmented image
                save_path = base_dir / f"{img_path.stem}_aug_{i}{img_path.suffix}"
                aug_img_pil.save(save_path)
        
        print("✅ Synthetic dataset created!")
        print(f"📊 Total images: {len(list(base_dir.glob('*.*')))}")
        
    except ImportError:
        print("❌ Albumentations not installed")
        print("💡 Run: pip install albumentations")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚂 Track Error Detection - Sample Data Setup")
    print("=" * 60)
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--synthetic':
        create_synthetic_dataset()
    else:
        download_sample_images()

