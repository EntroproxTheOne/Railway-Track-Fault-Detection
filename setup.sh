#!/bin/bash
# Track Error Detection - Setup Script

echo "🚂 Track Error Detection System Setup"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3.11 -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/annotations models checkpoints output
mkdir -p data/dataset/train data/dataset/valid data/dataset/test

# Create data.yaml
echo "📝 Creating data.yaml..."
python scripts/create_data_yaml.py

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Add your images to data/raw/"
echo "2. Run: python scripts/prepare_dataset.py"
echo "3. Train: python training/train.py"
echo "4. Detect: python inference/detect_realtime.py"

