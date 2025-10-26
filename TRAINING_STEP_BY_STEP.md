# 🚂 Step-by-Step Model Training Guide

## 📋 **Complete Training Process - Step by Step**

---

## **STEP 1: Activate Virtual Environment**

```bash
.\venv\Scripts\Activate
```

**What this does:**
- Activates Python 3.11 virtual environment
- Ensures you're using the right Python version with all packages installed

**Expected output:**
```
(venv) PS C:\Users\masoo\...\Track Error Detection>
```

---

## **STEP 2: Verify GPU Access**

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Expected output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
```

**If this fails:**
- GPU not detected - training will use CPU (slower)
- Check GPU drivers are installed

---

## **STEP 3: Check Dataset Configuration**

```bash
python scripts/create_data_yaml.py
```

**What this does:**
- Creates `data.yaml` configuration file
- This tells YOLO where to find your training data

**Expected output:**
```
[OK] Created data.yaml
[DATASET] Dataset root: ...\data\dataset
[CLASSES] Number of classes: 7
```

---

## **STEP 4: Verify Training Configuration**

```bash
cat config.yaml
```

**Key settings:**
- `model: yolov8n.pt` - Using nano model (best for your GPU)
- `img_size: 640` - Image resolution
- `batch_size: 16` - Batch size for training
- `epochs: 100` - Number of training iterations

---

## **STEP 5: START TRAINING**

```bash
python training/train.py --model nano --epochs 50
```

### **What Will Happen:**

1. **Loading Phase (30 seconds)**
   ```
   🚀 Starting Track Error Detection Training
   ============================================
   📦 Loading model: yolov8n.pt
   🔵 GPU: NVIDIA GeForce RTX 3060 Laptop GPU
   📊 Device: cuda
   ```

2. **Training Phase (30-60 minutes)**
   ```
   Epoch 1/50:   100%|████████████| 100/100 [00:45<00:00, 2.2it/s]
     Training loss: 1.234
     Validation mAP: 0.678
   
   Epoch 2/50:   100%|████████████| 100/100 [00:42<00:00, 2.3it/s]
     Training loss: 1.156
     Validation mAP: 0.701
   
   ... continues ...
   
   Epoch 50/50:   100%|████████████| 100/100 [00:40<00:00, 2.4it/s]
     Training loss: 0.456
     Validation mAP: 0.823
   ```

3. **Completion Phase (2-5 minutes)**
   ```
   ✅ Training completed!
   📊 Best mAP50: 0.823
   📊 Best mAP50-95: 0.734
   💾 Model saved to: models/track_error_model/weights/best.pt
   ```

### **Expected Timeline:**

| Phase | Duration |
|-------|----------|
| Setup | 1 minute |
| Training (50 epochs) | 30-60 minutes |
| Model Export | 2 minutes |
| **Total** | **35-65 minutes** |

---

## **STEP 6: Monitor Training (Optional)**

**In another terminal:**

```bash
# Monitor GPU usage
nvidia-smi -l 1
```

**You'll see:**
```
GPU-Util: 85-95%
Memory: 4000 MB / 6144 MB
Temp: 65°C
```

**Normal values:**
- GPU Utilization: 80-95% (good!)
- Memory: 4-5 GB / 6 GB (normal)
- Temperature: 50-70°C (safe)

---

## **STEP 7: Check Training Results**

After training completes:

```bash
# View the saved model
dir models\track_error_model\weights\

# You should see:
# - best.pt (best model)
# - last.pt (final epoch model)
```

---

## **STEP 8: Test the Trained Model**

```bash
# Start real-time detection
python inference/detect_realtime.py --model models/track_error_model/weights/best.pt
```

**What you'll see:**
- Camera window opens
- Detections appear on screen
- Bounding boxes around objects
- Labels and confidence scores

**Press 'q' to quit**

---

## 📊 **Training Progress Indicators**

### **Good Signs:**
- ✅ Loss decreasing over time
- ✅ mAP increasing
- ✅ GPU usage 80-95%
- ✅ No error messages
- ✅ Training completes successfully

### **Bad Signs:**
- ❌ Loss increasing (overfitting)
- ❌ mAP decreasing
- ❌ Out of memory errors
- ❌ Training crashes

---

## 🎯 **What Model Does After Training:**

### **With Transfer Learning (Your Case):**
- ✅ Detects 80+ object types (COCO dataset)
- ✅ Works on any objects initially
- ✅ General purpose detection
- ✅ Proves system works

### **After Adding Track Data:**
- ✅ Detects track-specific defects
- ✅ Recognizes Simple/Moderate/Severe errors
- ✅ Specialized for railway monitoring

---

## 💡 **Pro Tips:**

1. **Let it run!** - Don't interrupt training
2. **Monitor GPU temp** - Keep under 80°C
3. **Check progress** - Loss should decrease
4. **Save power** - Training uses lots of electricity
5. **Be patient** - Quality takes time

---

## ⚡ **Quick Start (All Steps in Order):**

```bash
# 1. Activate environment
.\venv\Scripts\Activate

# 2. Create config
python scripts/create_data_yaml.py

# 3. START TRAINING
python training/train.py --model nano --epochs 50

# 4. After training completes, test it
python inference/detect_realtime.py --model models/track_error_model/weights/best.pt
```

---

## 🎉 **Ready to Start?**

I'll run these commands for you when you say "yes"!

**Or follow along manually using the steps above.**

**Time required:** ~45 minutes  
**GPU will be busy:** Yes  
**Can use computer:** Yes (but keep it plugged in)

