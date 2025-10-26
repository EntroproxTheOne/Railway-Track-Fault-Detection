"""
Real-time Track Error Detection
Detects track errors in real-time using camera feed
"""

import cv2
import torch
import yaml
from pathlib import Path
import argparse
import numpy as np
from typing import Dict, Tuple, List
import time

class TrackErrorDetector:
    def __init__(self, model_path: str, config_path: str = 'config.yaml'):
        """
        Initialize track error detector.
        
        Args:
            model_path: Path to trained YOLOv8 model
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        
        # Get severity mappings
        self.severity_mapping = self.config.get('severity_mapping', {})
        
        # Get colors for severity levels
        self.severity_colors = self.config.get('severity_colors', {
            'simple': (0, 255, 0),      # Green
            'moderate': (0, 165, 255),  # Orange
            'severe': (0, 0, 255)        # Red
        })
        
        # Detection settings
        self.conf_threshold = self.config.get('confidence_threshold', 0.45)
        self.iou_threshold = self.config.get('iou_threshold', 0.5)
        
        # Map class names to severity
        self.class_to_severity = {}
        for severity, classes in self.severity_mapping.items():
            for class_name in classes:
                self.class_to_severity[class_name] = severity
        
        print(f"✅ Model loaded: {model_path}")
        print(f"🎯 Confidence threshold: {self.conf_threshold}")
    
    def get_severity(self, class_name: str) -> str:
        """Get severity level for a class."""
        return self.class_to_severity.get(class_name, 'simple')
    
    def draw_detection(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw detection on frame with severity color coding.
        
        Args:
            frame: Input frame
            detection: Detection result from YOLOv8
            
        Returns:
            Annotated frame
        """
        boxes = detection.boxes
        
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = self.model.names[cls]
            
            # Get severity and color
            severity = self.get_severity(class_name)
            color = self.severity_colors.get(severity, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name} {severity.title()}"
            score = f"{conf:.2f}"
            label_text = f"{label} {score}"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_height = label_size[1] + 10
            cv2.rectangle(frame, (x1, y1 - label_height), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process single frame and return annotated frame with detections.
        
        Args:
            frame: Input frame
            
        Returns:
            Annotated frame and detection results
        """
        # Run detection
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # Draw detections
        annotated_frame = results[0].plot()
        
        # Get detailed detections for info
        detections_info = {
            'count': len(results[0].boxes),
            'classes': [self.model.names[int(box.cls)] for box in results[0].boxes]
        }
        
        return annotated_frame, detections_info

def create_camera(config: Dict):
    """Create camera object based on configuration."""
    camera_settings = config.get('camera', {})
    
    # Try to open camera
    cap = cv2.VideoCapture(camera_settings.get('device_id', 0))
    
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open camera")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_settings.get('width', 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_settings.get('height', 720))
    cap.set(cv2.CAP_PROP_FPS, camera_settings.get('fps', 30))
    
    return cap

def main():
    """Main real-time detection loop."""
    parser = argparse.ArgumentParser(description='Real-time Track Error Detection')
    parser.add_argument('--model', type=str, default='models/track_error_model/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--webcam', type=int, default=0,
                       help='Webcam device ID')
    parser.add_argument('--record', action='store_true',
                       help='Record video with detections')
    parser.add_argument('--save-annotations', action='store_true',
                       help='Save frames with annotations')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("💡 Train a model first using: python training/train.py")
        return
    
    # Load detector
    detector = TrackErrorDetector(args.model, args.config)
    
    # Load config for camera settings
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create camera
    try:
        cap = create_camera(config)
        print(f"📷 Camera opened successfully")
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return
    
    # Setup video writer if recording
    out = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output/detections.mp4', fourcc, 30.0, (1280, 720))
        print("📹 Recording enabled")
    
    # FPS tracking
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    print("\n🎥 Starting real-time detection...")
    print("Press 'q' to quit")
    print("=" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Process frame
            annotated_frame, detections = detector.process_frame(frame)
            
            # Draw FPS
            cv2.putText(annotated_frame, f'FPS: {fps:.1f}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw detection count
            cv2.putText(annotated_frame, f'Detections: {detections["count"]}',
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Track Error Detection', annotated_frame)
            
            # Write frame if recording
            if out is not None:
                out.write(annotated_frame)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter == 30:
                fps = 30 / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
                print(f"📊 FPS: {fps:.1f} | Detections: {detections['count']}")
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping detection...")
    
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("✅ Detection stopped")

if __name__ == "__main__":
    main()

