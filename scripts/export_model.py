"""
Model Export Script
Export trained models in different formats for deployment
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import argparse

def export_to_onnx(model_path: str, output_dir: str = 'models'):
    """Export model to ONNX format (optimal for Pi)."""
    print(f"📦 Exporting {model_path} to ONNX...")
    
    model = YOLO(model_path)
    
    # Export to ONNX with optimizations
    output_path = model.export(
        format='onnx',
        dynamic=True,  # Allow dynamic batch sizes
        simplify=True,  # Simplify ONNX graph
        opset=12  # ONNX opset version
    )
    
    print(f"✅ ONNX model saved: {output_path}")
    return output_path

def export_to_tflite(model_path: str, output_dir: str = 'models'):
    """Export model to TensorFlow Lite format."""
    print(f"📦 Exporting {model_path} to TensorFlow Lite...")
    
    model = YOLO(model_path)
    
    # Export to TensorFlow Lite
    output_path = model.export(format='tflite')
    
    print(f"✅ TensorFlow Lite model saved: {output_path}")
    return output_path

def export_to_torchscript(model_path: str, output_dir: str = 'models'):
    """Export model to TorchScript format."""
    print(f"📦 Exporting {model_path} to TorchScript...")
    
    model = YOLO(model_path)
    
    # Export to TorchScript
    output_path = model.export(format='torchscript')
    
    print(f"✅ TorchScript model saved: {output_path}")
    return output_path

def export_to_engine(model_path: str, output_dir: str = 'models'):
    """Export model to TensorRT engine (requires GPU)."""
    print(f"📦 Exporting {model_path} to TensorRT engine...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Skipping TensorRT export.")
        return None
    
    model = YOLO(model_path)
    
    # Export to TensorRT
    output_path = model.export(format='engine', half=True)
    
    print(f"✅ TensorRT engine saved: {output_path}")
    return output_path

def export_all(model_path: str):
    """Export model to all formats."""
    print("🚀 Exporting model to all formats...")
    print("=" * 60)
    
    exports = {
        'ONNX': export_to_onnx,
        'TensorFlow Lite': export_to_tflite,
        'TorchScript': export_to_torchscript,
        'TensorRT': export_to_engine,
    }
    
    results = {}
    for format_name, export_func in exports.items():
        try:
            output_path = export_func(model_path)
            results[format_name] = 'Success'
        except Exception as e:
            print(f"❌ {format_name} export failed: {str(e)}")
            results[format_name] = 'Failed'
    
    print("\n📊 Export Summary:")
    print("=" * 60)
    for format_name, status in results.items():
        print(f"{format_name}: {status}")

def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8 Model')
    parser.add_argument('--model', type=str, 
                       default='models/track_error_model/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--format', type=str, 
                       choices=['onnx', 'tflite', 'torchscript', 'engine', 'all'],
                       default='all',
                       help='Export format')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("💡 Train a model first using: python training/train.py")
        return
    
    # Export based on format
    if args.format == 'all':
        export_all(args.model)
    elif args.format == 'onnx':
        export_to_onnx(args.model)
    elif args.format == 'tflite':
        export_to_tflite(args.model)
    elif args.format == 'torchscript':
        export_to_torchscript(args.model)
    elif args.format == 'engine':
        export_to_engine(args.model)

if __name__ == "__main__":
    main()

