# 🚀 Quick Start Guide - Track Error Detection

## ✅ Setup Complete!

Your environment is ready. Here's what's next:

### 📋 What's Been Created

```
✅ Python 3.11 virtual environment (venv/)
✅ Project structure (data/, training/, inference/, scripts/)
✅ Configuration files (config.yaml, requirements.txt)
✅ Training script (training/train.py)
✅ Real-time detection script (inference/detect_realtime.py)
✅ Dataset preparation tools (scripts/)
✅ Model export utilities (scripts/export_model.py)
✅ Comprehensive README with dataset sources
```

---

## 🎯 Next Steps

### Step 1: Get Your Dataset

**Option A: Download from Public Sources**
```bash
# Check out these sources:
1. Kaggle: https://www.kaggle.com/datasets
   Search: "railway track", "railway defect"
   
2. Google Dataset Search:
   https://datasetsearch.research.google.com/
   
3. Roboflow Universe:
   https://universe.roboflow.com/

# Or use our helper script
python scripts/download_sample_data.py
```

**Option B: Use Your Own Images**
```bash
# 1. Place images in data/raw/ with severity in folder names
data/raw/
├── simple/
│   └── missing_bolt_001.jpg
├── moderate/
│   └── ballast_issue_001.jpg
└── severe/
    └── track_crack_001.jpg

# 2. Prepare dataset
python scripts/prepare_dataset.py

# 3. Create YOLO config
python scripts/create_data_yaml.py
```

### Step 2: Train Your Model

```bash
# Train with nano model (best for Pi)
python training/train.py --model nano

# This will:
# ✅ Train YOLOv8-nano on your data
# ✅ Save best model to models/track_error_model/weights/best.pt
# ✅ Export to ONNX for Pi deployment
# ✅ Show training metrics
```

**Expected Training Time:**
- Desktop with GPU: 1-2 hours
- Desktop CPU: 4-6 hours
- Raspberry Pi (not recommended for training)

### Step 3: Run Real-Time Detection

```bash
# Basic detection with webcam
python inference/detect_realtime.py

# With specific model
python inference/detect_realtime.py --model models/track_error_model/weights/best.pt

# Record detections
python inference/detect_realtime.py --record
```

**Controls:**
- Press `'q'` to quit
- Press `'r'` to record (if --record flag is used)

---

## 🥧 Deploy on Raspberry Pi

### 1. Train on Desktop/Cloud
```bash
python training/train.py --model nano
```

### 2. Export for Pi
```bash
python scripts/export_model.py --format onnx
```

### 3. Copy to Pi
```bash
# From your desktop
scp models/track_error_model/weights/best.onnx pi@raspberrypi:/home/pi/track_detection/
```

### 4. Run on Pi
```bash
# SSH to Pi
ssh pi@raspberrypi

# Navigate to project
cd track_detection

# Run detection
python inference/detect_realtime.py --model best.onnx
```

---

## 📊 What Gets Detected?

| Error Type | Severity | Color Code |
|-----------|----------|------------|
| Track Crack | 🟥 Severe | Red |
| Broken Sleeper | 🟥 Severe | Red |
| Gauge Error | 🟥 Severe | Red |
| Ballast Issue | 🟧 Moderate | Orange |
| Fishplate Issue | 🟧 Moderate | Orange |
| Missing Bolt | 🟩 Simple | Green |
| Crushed Stone | 🟩 Simple | Green |

---

## 🎨 Expected Output

Your detection will look like this:

```
┌─────────────────────────────────────┐
│  📹 Camera Feed                      │
│                                      │
│  [Boundary Box]                      │
│  Track Crack [Severe] 0.87           │
│                                      │
│  FPS: 12.5                           │
│  Detections: 3                       │
└─────────────────────────────────────┘
```

- **Color-coded boxes** by severity
- **Confidence scores** for each detection
- **Real-time FPS** counter
- **Detection count** overlay

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Detection sensitivity
confidence_threshold: 0.45  # Lower = more detections

# Camera settings
camera:
  device_id: 0
  width: 1280
  height: 720

# Pi optimization
pi_mode: True
pi_img_size: 416  # Smaller = faster
```

---

## 🛠️ Troubleshooting

### No Images Found
```bash
# Check: ls data/raw/
# If empty, add images or download from Kaggle
```

### Model Not Found
```bash
# You need to train first
python training/train.py --model nano
```

### Low FPS on Pi
```yaml
# Edit config.yaml
pi_img_size: 416  # Reduce from 640
confidence_threshold: 0.5  # Reduce false positives
```

### Camera Not Working
```bash
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

---

## 📚 Additional Resources

- **Full Documentation**: See `README.md`
- **Dataset Sources**: See README.md section "Dataset Sources"
- **Model Sizes**: nano (fast), small (balanced), medium/large (accurate but slow)
- **Augmentation**: Use `scripts/download_sample_data.py --synthetic`

---

## 🎉 You're All Set!

Run your first detection:

```bash
# 1. Activate venv (if not active)
.\venv\Scripts\Activate  # Windows
source venv/bin/activate  # Linux

# 2. Add images to data/raw/
# 3. Prepare dataset
python scripts/prepare_dataset.py
python scripts/create_data_yaml.py

# 4. Train
python training/train.py --model nano

# 5. Detect!
python inference/detect_realtime.py
```

**Happy Detecting! 🚂✨**

