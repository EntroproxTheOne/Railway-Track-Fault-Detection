# 📦 Dataset Setup Guide - Track Error Detection

## 🎯 Overview

This guide will walk you through getting and setting up your dataset for training the track error detection model.

---

## 📋 Quick Summary (3 Steps)

1. **Download/Add Images** → Place in `data/raw/`
2. **Prepare Dataset** → Run organization script
3. **Ready to Train** → Start training

---

## 🚀 Method 1: Using Rail-5k Dataset (RECOMMENDED!)

### Step 1: Download Rail-5k Dataset

**This is a real railway dataset with 5000+ images!**

```bash
# Automatic download (requires git)
python scripts/download_rail5k.py

# Or manual download instructions
python scripts/download_rail5k.py  # Will show instructions if git fails
```

**Manual Download:**
1. Visit: https://github.com/TommyZihao/Rail-5k-dataset
2. Click "Code" → "Download ZIP"
3. Extract to `data/raw/rail-5k-temp/`
4. Run: `python scripts/download_rail5k.py --prepare-only`

This dataset includes:
- Real railway track images
- Annotations for object detection
- Large dataset (5000+ images)

---

## 🎯 Method 2: Using Public Datasets

### Step 1: Find Public Datasets

**Option A: Kaggle**
```bash
# 1. Visit kaggle.com
# 2. Search for: "railway track", "railway defect", "railway anomaly"
# 3. Download dataset
# 4. Extract to data/raw/
```

**Option B: Google Dataset Search**
```bash
# 1. Visit: https://datasetsearch.research.google.com/
# 2. Search: "railway infrastructure detection"
# 3. Download and extract to data/raw/
```

**Option C: Roboflow Universe**
```bash
# 1. Visit: https://universe.roboflow.com/
# 2. Search: "railway" or "train track"
# 3. Export in YOLOv8 format
```

### Step 2: Organize Images

If you downloaded a dataset that isn't organized by severity:

```bash
# Create severity folders
mkdir data/raw/simple
mkdir data/raw/moderate
mkdir data/raw/severe

# Move or copy images to appropriate folders based on their severity
# Examples:
# - Missing bolts, crushed stones → data/raw/simple/
# - Ballast issues, fishplate problems → data/raw/moderate/
# - Track cracks, broken sleepers → data/raw/severe/
```

### Step 3: Prepare Dataset

```bash
# Activate virtual environment (if not active)
.\venv\Scripts\Activate

# Run dataset preparation script
python scripts/prepare_dataset.py
```

**What this does:**
- Organizes images into train/valid/test splits (70/20/10)
- Separates by severity (simple/moderate/severe)
- Validates the dataset structure
- Shows statistics

### Step 4: Create YOLO Configuration

```bash
# Create data.yaml for training
python scripts/create_data_yaml.py
```

**Output:** `data.yaml` file that tells YOLO where your data is

---

## 🎨 Method 2: Create Your Own Dataset

### Step 1: Collect Images

**Where to get images:**
1. Take photos at railway stations/tracks
2. Search Google Images: "railway track crack", "railway track defect"
3. Use your own railway monitoring photos
4. Ask Indian Railways for training data

**Recommended:**
- At least 100-200 images per class (or per severity level)
- Various lighting conditions (day, night, different weather)
- Different angles and distances
- Mix of close-up and wide shots

### Step 2: Organize by Severity

```bash
# Your structure should look like:
data/raw/
├── simple/
│   ├── missing_bolt_001.jpg
│   ├── missing_bolt_002.jpg
│   ├── crushed_stone_001.jpg
│   └── ...
├── moderate/
│   ├── ballast_issue_001.jpg
│   ├── fishplate_001.jpg
│   └── ...
└── severe/
    ├── track_crack_001.jpg
    ├── broken_sleeper_001.jpg
    ├── gauge_error_001.jpg
    └── ...
```

**Severity Classification:**
- **Simple (🟩)**: Missing bolts, crushed stones, minor issues
- **Moderate (🟧)**: Ballast problems, fishplate issues, medium severity
- **Severe (🟥)**: Cracks, broken sleepers, gauge errors, critical issues

### Step 3: Annotate Images (Optional - For Precise Training)

If you want best accuracy, manually annotate images:

**Using LabelImg (Recommended):**

```bash
# Install LabelImg
pip install labelImg

# Run LabelImg
labelImg

# In LabelImg:
# 1. Open Folder: data/raw/
# 2. Change save dir to: data/annotations/
# 3. Use YOLO format
# 4. Start annotating:
#    - Draw boxes around track errors
#    - Label them with class names:
#      - track_crack
#      - missing_bolt
#      - ballast_issue
#      - broken_sleeper
#      - gauge_error
#      - fishplate_issue
#      - crushed_stone
```

### Step 4: Prepare Dataset

```bash
# Run the preparation script
python scripts/prepare_dataset.py
```

This will:
- Split data: 70% train, 20% validation, 10% test
- Organize by severity
- Validate structure
- Print statistics

---

## 🎲 Method 3: Synthetic Data Augmentation (Expand Small Datasets)

If you have only a few images, augment them:

```bash
# Run synthetic data generation
python scripts/download_sample_data.py --synthetic
```

**What this does:**
- Applies transformations (rotation, brightness, blur, noise)
- Creates 5x variations of each image
- Helps with small datasets

---

## 📊 Verify Your Dataset

After preparation, check your dataset:

```bash
# Check structure
tree data/dataset  # Or: Get-ChildItem -Recurse data/dataset
```

**Expected structure:**
```
data/dataset/
├── train/
│   ├── simple/     (70% of simple images)
│   ├── moderate/
│   └── severe/
├── valid/
│   ├── simple/     (20% of images)
│   ├── moderate/
│   └── severe/
└── test/
    ├── simple/     (10% of images)
    ├── moderate/
    └── severe/
```

---

## ✅ Dataset Checklist

Before training, verify:

- [ ] Images in `data/raw/` (minimum 50-100 total)
- [ ] Organized by severity (simple/moderate/severe)
- [ ] Run `python scripts/prepare_dataset.py` ✅
- [ ] Run `python scripts/create_data_yaml.py` ✅
- [ ] `data.yaml` file exists
- [ ] Check statistics shown by preparation script

---

## 🚨 Common Issues & Solutions

### Issue 1: "No images found"
```bash
# Problem: data/raw/ is empty
# Solution: Add images first!

# Quick test with sample images:
# 1. Download any railway track images from web
# 2. Save to data/raw/simple/ (or moderate/severe)
# 3. Re-run: python scripts/prepare_dataset.py
```

### Issue 2: "Dataset validation failed"
```bash
# Problem: Missing folders or empty directories
# Solution: 
mkdir -p data/dataset/train/simple
mkdir -p data/dataset/train/moderate
mkdir -p data/dataset/train/severe
mkdir -p data/dataset/valid/simple
mkdir -p data/dataset/valid/moderate
mkdir -p data/dataset/valid/severe
# Add images, then re-run
```

### Issue 3: Very few images (< 50)
```bash
# Solution: Use augmentation
python scripts/download_sample_data.py --synthetic

# Or: Use transfer learning from pre-trained model (YOLO does this automatically)
```

---

## 📈 Minimal Dataset Requirements

| Severity | Minimum Images | Recommended |
|----------|----------------|-------------|
| Simple | 20-30 | 100+ |
| Moderate | 20-30 | 100+ |
| Severe | 20-30 | 100+ |
| **Total** | **60-90** | **300+** |

**Note:** You can start with less and add more later. YOLO will use transfer learning from COCO dataset.

---

## 🎯 What's Next?

Once your dataset is ready:

1. **Start Training:**
   ```bash
   python training/train.py --model nano
   ```

2. **Monitor Training:**
   - Watch for mAP scores
   - Check loss values decreasing
   - Training takes 1-2 hours on GPU

3. **Test Model:**
   ```bash
   python inference/detect_realtime.py --model models/track_error_model/weights/best.pt
   ```

---

## 💡 Pro Tips

1. **Start Small**: Begin with 50-100 images, test, then add more
2. **Mix Sources**: Combine your photos with public datasets
3. **Quality > Quantity**: Better to have 100 good images than 1000 bad ones
4. **Consistent Annotation**: Use same labeling style if annotating manually
5. **Track Sources**: Keep track of where images came from

---

## 📞 Need Help?

- Check `README.md` for comprehensive documentation
- See `QUICK_START.md` for quick reference
- Dataset sources are listed in `README.md` section "Dataset Sources"

**Ready to train!** 🚂✨

