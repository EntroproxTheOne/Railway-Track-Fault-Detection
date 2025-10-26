# 🚂 Track Error Detection System - Setup Complete!

## ✅ What Has Been Created

### 1. **Virtual Environment** ✅
- Python 3.11 virtual environment created in `venv/`
- All dependencies installed (ultralytics, opencv-python, torch, etc.)
- Ready for training and inference

### 2. **Project Structure** ✅
```
Track Error Detection/
├── data/                    # Dataset organization
│   ├── raw/                # Place your images here
│   ├── dataset/            # Organized train/val/test splits
│   └── annotations/        # YOLO labels
├── models/                 # Trained model storage
├── training/               # Training scripts
│   ├── train.py           # Main training script
│   └── config.yaml        # Model configuration
├── inference/             # Real-time detection
│   └── detect_realtime.py # Camera-based detection
├── scripts/              # Utility scripts
│   ├── prepare_dataset.py # Dataset organizer
│   ├── create_data_yaml.py # YOLO config generator
│   ├── export_model.py    # Model exporter (ONNX, TFLite)
│   └── download_sample_data.py # Dataset downloader
├── utils/                 # Helper functions
│   └── visualization.py  # Drawing utilities
├── config.yaml           # Main configuration
├── data.yaml             # YOLO dataset config
├── requirements.txt      # Python dependencies
├── README.md             # Full documentation
└── QUICK_START.md        # Quick start guide
```

### 3. **Key Features** ✅

**Real-Time Detection**
- Live camera feed processing
- Color-coded bounding boxes (Green/Orange/Red by severity)
- Confidence scores displayed
- FPS counter and detection statistics

**Raspberry Pi Optimized**
- YOLOv8-nano model for speed
- ONNX export for efficient inference
- Configurable image size (416x416 for Pi)
- Expected 8-12 FPS on Pi 4

**Severity Classification**
- **Simple** (Green): Missing bolts, crushed stones
- **Moderate** (Orange): Ballast issues, fishplate problems
- **Severity** (Red): Track cracks, broken sleepers, gauge errors

---

## 🎯 Next Steps

### **STEP 1: Add Your Dataset**

You have two options:

**Option A: Use Public Dataset**
```bash
# Visit these sources:
1. Kaggle.com → Search "railway track defect"
2. Google Dataset Search → Search "railway infrastructure"
3. Roboflow Universe → Search "rail track"

# Or use synthetic augmentation
python scripts/download_sample_data.py --synthetic
```

**Option B: Use Your Own Images**
```bash
# 1. Create folders in data/raw/
mkdir data/raw/simple
mkdir data/raw/moderate  
mkdir data/raw/severe

# 2. Add your images
# Place images in appropriate severity folders

# 3. Organize dataset
python scripts/prepare_dataset.py
```

### **STEP 2: Train the Model**

```bash
# Activate virtual environment (if not active)
.\venv\Scripts\Activate

# Start training with nano model (best for Pi)
python training/train.py --model nano

# Training will:
# - Train for 100 epochs
# - Save best model to models/track_error_model/weights/best.pt
# - Export ONNX model for Pi
# - Show mAP metrics
```

**Training Time:**
- GPU: ~1-2 hours
- CPU: ~4-6 hours

### **STEP 3: Run Detection**

```bash
# Start real-time detection
python inference/detect_realtime.py --model models/track_error_model/weights/best.pt
```

**Controls:**
- Press `'q'` to quit
- Press `'r'` to record video

---

## 🥧 Raspberry Pi Deployment

### **Desktop → Pi Workflow:**

1. **Train on Desktop:**
   ```bash
   python training/train.py --model nano
   ```

2. **Export for Pi:**
   ```bash
   python scripts/export_model.py --format onnx
   ```

3. **Copy to Pi:**
   ```bash
   scp models/track_error_model/weights/best.onnx pi@192.168.x.x:/home/pi/
   ```

4. **Run on Pi:**
   ```bash
   ssh pi@192.168.x.x
   python inference/detect_realtime.py --model best.onnx
   ```

---

## 📊 What You'll See

The detection system will display:

```
┌──────────────────────────────────┐
│ Webcam Feed                       │
│                                   │
│  ┌────────────┐                   │
│  │ [Box]      │  Track Crack      │
│  │ Severe 0.87│  [Red Box]        │
│  │            │                   │
│  └────────────┘                   │
│                                   │
│ FPS: 12.5                         │
│ Detections: 3                     │
│                                   │
└──────────────────────────────────┘
```

- **Green boxes** = Simple errors (missing bolt)
- **Orange boxes** = Moderate errors (ballast issue)
- **Red boxes** = Severe errors (crack, broken sleeper)

---

## 🎨 Track Error Classes

Your model detects 7 classes:

| # | Class Name | Severity | Description |
|---|-----------|----------|-------------|
| 0 | track_crack | Severe | Visible cracks in rails |
| 1 | missing_bolt | Simple | Loose or missing bolts |
| 2 | ballast_issue | Moderate | Degraded ballast stones |
| 3 | broken_sleeper | Severe | Damaged or missing sleepers |
| 4 | gauge_error | Severe | Incorrect track width |
| 5 | fishplate_issue | Moderate | Damaged joint plates |
| 6 | crushed_stone | Simple | Crushed ballast stones |

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Detection sensitivity
confidence_threshold: 0.45  # Lower = more detections

# Camera settings
camera:
  device_id: 0       # Camera index
  width: 1280
  height: 720
  fps: 30

# Pi optimization  
pi_mode: False      # Set True for Pi
pi_img_size: 416    # Smaller = faster
```

---

## 🛠️ Troubleshooting

### **"Model not found" error**
```bash
# You need to train first!
python training/train.py --model nano
```

### **"No images found"**
```bash
# Add images to data/raw/
# Then run:
python scripts/prepare_dataset.py
```

### **Low FPS on Pi**
```yaml
# Edit config.yaml
pi_img_size: 416  # Reduce from 640
confidence_threshold: 0.5  # Increase to reduce processing
```

### **Camera not working**
```bash
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

---

## 📚 Documentation

- **README.md** - Full documentation with dataset sources
- **QUICK_START.md** - Step-by-step quick start guide
- **config.yaml** - All configuration options
- **training/train.py** - Training script with options

---

## 🎓 Learning Path

1. **Week 1**: Collect/Download dataset, train basic model
2. **Week 2**: Fine-tune on your specific track data
3. **Week 3**: Deploy to Raspberry Pi, optimize settings
4. **Week 4**: Test in real railway environment

---

## 📈 Expected Performance

| Platform | Model | FPS | mAP |
|----------|-------|-----|-----|
| Desktop (GPU) | YOLOv8-nano | 45+ | 0.65+ |
| Desktop (CPU) | YOLOv8-nano | 12-15 | 0.65+ |
| Raspberry Pi 4 | YOLOv8-nano | 8-12 | 0.60+ |
| Raspberry Pi 5 | YOLOv8-nano | 15-20 | 0.60+ |

---

## 🎉 You're All Set!

**Quick Test:**
```bash
# 1. Add sample images to data/raw/
# 2. Prepare dataset
python scripts/prepare_dataset.py

# 3. Train
python training/train.py --model nano

# 4. Detect!
python inference/detect_realtime.py
```

**Questions?** Check the comprehensive README.md or QUICK_START.md!

---

## 🙏 Resources

- **Dataset Sources**: See README.md "Dataset Sources" section
- **YOLOv8 Docs**: https://docs.ultralytics.com
- **OpenCV Docs**: https://docs.opencv.org
- **Indian Railways**: https://indianrailways.gov.in

---

**Made with ❤️ for Railway Safety** 🚂✨

