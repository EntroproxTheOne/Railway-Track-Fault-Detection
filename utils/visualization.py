"""
Visualization Utilities for Track Error Detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

def draw_severity_box(image: np.ndarray, 
                     box: Tuple[int, int, int, int],
                     class_name: str,
                     severity: str,
                     confidence: float,
                     class_colors: Dict = None) -> np.ndarray:
    """
    Draw detection box with severity-based color coding.
    
    Args:
        image: Input image
        box: Bounding box (x1, y1, x2, y2)
        class_name: Name of detected class
        severity: Severity level
        confidence: Confidence score
        class_colors: Color mapping for classes
        
    Returns:
        Annotated image
    """
    x1, y1, x2, y2 = box
    
    # Color mapping based on severity
    severity_colors = {
        'simple': (0, 255, 0),       # Green
        'moderate': (0, 165, 255),   # Orange
        'severe': (0, 0, 255)        # Red
    }
    
    color = severity_colors.get(severity, (255, 255, 255))
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    
    # Create label
    label = f"{class_name} [{severity}]"
    score = f"{confidence:.2f}"
    label_text = f"{label} {score}"
    
    # Get label size
    (label_width, label_height), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    
    # Draw label background
    cv2.rectangle(image, 
                 (x1, y1 - label_height - 10 - baseline),
                 (x1 + label_width, y1),
                 color, -1)
    
    # Draw label text
    cv2.putText(image, label_text,
               (x1, y1 - 5 - baseline),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
               (255, 255, 255), 2)
    
    return image

def draw_fps(image: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter on image."""
    cv2.putText(image, f'FPS: {fps:.1f}',
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1, (0, 255, 0), 2)
    return image

def draw_stats(image: np.ndarray, 
              total_detections: int,
              detections_by_severity: Dict) -> np.ndarray:
    """Draw detection statistics on image."""
    y_offset = 70
    line_height = 30
    
    # Total detections
    cv2.putText(image, f'Total Detections: {total_detections}',
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
               1, (0, 255, 0), 2)
    
    # Severity breakdown
    y_offset += line_height
    for severity, count in detections_by_severity.items():
        color = (0, 255, 0) if severity == 'simple' else \
               (0, 165, 255) if severity == 'moderate' else (0, 0, 255)
        cv2.putText(image, f'{severity.capitalize()}: {count}',
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, color, 2)
        y_offset += line_height
    
    return image

def create_summary_image(detections: List[Dict],
                        stats: Dict) -> np.ndarray:
    """
    Create summary visualization of all detections.
    
    Args:
        detections: List of detection dictionaries
        stats: Statistics dictionary
        
    Returns:
        Summary image
    """
    # Create blank image for summary
    summary = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw title
    cv2.putText(summary, 'Track Error Detection Summary',
               (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
               1.2, (0, 0, 0), 2)
    
    # Draw statistics
    y_offset = 100
    for key, value in stats.items():
        cv2.putText(summary, f'{key}: {value}',
                   (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 0, 0), 2)
        y_offset += 40
    
    return summary

