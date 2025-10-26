"""
Track Error Detection - Training Script
Trains YOLOv8 model on your track error dataset
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch
import os

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(config_path='config.yaml', model_size='nano'):
    """
    Train YOLOv8 model for track error detection.
    
    Args:
        config_path: Path to configuration YAML file
        model_size: Model size ('nano', 'small', 'medium', 'large')
    """
    
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Training on: {device}")
    
    # Initialize model
    model_name = f'yolov8{model_size[0]}.pt'  # yolov8n.pt, yolov8s.pt, etc.
    print(f"📦 Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Check if data.yaml exists
    if not Path('data.yaml').exists():
        print("❌ data.yaml not found! Please run scripts/create_data_yaml.py first")
        return
    
    # Training parameters
    training_params = {
        'data': 'data.yaml',
        'epochs': config.get('epochs', 100),
        'imgsz': config.get('img_size', 640),
        'batch': config.get('batch_size', 16),
        'device': device,
        'project': 'models',
        'name': 'track_error_model',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'lr0': config.get('lr0', 0.001),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'warmup_epochs': config.get('warmup_epochs', 3.0),
        'warmup_momentum': config.get('warmup_momentum', 0.8),
        'warmup_bias_lr': config.get('warmup_bias_lr', 0.1),
    }
    
    # Pi optimization mode
    if config.get('pi_mode', False):
        training_params['imgsz'] = config.get('pi_img_size', 416)
        print(f"🥧 Pi optimization mode: Using image size {training_params['imgsz']}")
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    # Train the model
    results = model.train(**training_params)
    
    print("\n✅ Training completed!")
    print("=" * 60)
    
    # Print results summary
    print("\n📊 Training Results:")
    print(f"   Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    # Export models for different deployment
    export_models(model, results)
    
    return model, results

def export_models(model, results):
    """Export models in different formats for deployment."""
    print("\n📦 Exporting models for deployment...")
    
    export_formats = ['onnx', 'tflite', 'engine', 'torchscript']
    
    for fmt in export_formats:
        try:
            if fmt == 'onnx':
                model.export(format='onnx', dynamic=True, simplify=True)
                print(f"   ✅ ONNX model exported (Pi optimized)")
            elif fmt == 'tflite':
                model.export(format='tflite')
                print(f"   ✅ TensorFlow Lite model exported")
            elif fmt == 'engine':
                model.export(format='engine', half=True)
                print(f"   ✅ TensorRT engine exported")
            elif fmt == 'torchscript':
                model.export(format='torchscript')
                print(f"   ✅ TorchScript model exported")
        except Exception as e:
            print(f"   ⚠️  Could not export {fmt}: {str(e)}")
    
    print("\n✅ Model export completed!")

def validate_model(model_path):
    """Validate trained model."""
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(data='data.yaml')
    
    print("\n📊 Validation Results:")
    print(f"   mAP50: {metrics.box.map50}")
    print(f"   mAP50-95: {metrics.box.map}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Track Error Detection Model')
    parser.add_argument('--model', type=str, default='nano', 
                       choices=['nano', 'small', 'medium', 'large'],
                       help='Model size (default: nano for Pi)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--validate', type=str, default=None,
                       help='Path to model to validate')
    
    args = parser.parse_args()
    
    if args.validate:
        print(f"🔍 Validating model: {args.validate}")
        validate_model(args.validate)
    else:
        print("🚀 Starting Track Error Detection Training")
        print("=" * 60)
        train_model(args.config, args.model)

