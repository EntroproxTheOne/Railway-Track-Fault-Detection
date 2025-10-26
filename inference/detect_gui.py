"""
GUI App for Real-Time Track Error Detection
Simple tkinter-based interface with webcam display
"""

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import yaml
from ultralytics import YOLO
import threading
import time

class TrackDetectorGUI:
    def __init__(self, model_path='models/track_error_model/weights/best.pt'):
        self.root = tk.Tk()
        self.root.title("🚂 Track Error Detection System")
        self.root.geometry("1280x800")
        self.root.configure(bg='#2b2b2b')
        
        # Load model
        print("Loading model...")
        self.model = YOLO(model_path)
        
        # Load config
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Camera settings
        self.cap = None
        self.running = False
        self.current_frame = None
        self.fps = 0
        self.detection_count = 0
        
        # Severity colors (BGR to RGB for display)
        self.severity_colors = {
            'simple': '#00FF00',    # Green
            'moderate': '#FFA500',  # Orange
            'severe': '#FF0000'     # Red
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the GUI interface"""
        # Header
        header = tk.Frame(self.root, bg='#1a1a1a', height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        
        title = tk.Label(header, text="🚂 Railway Track Error Detection", 
                        font=('Arial', 20, 'bold'), bg='#1a1a1a', fg='white')
        title.pack(pady=10)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video frame (left side)
        video_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.RAISED, borderwidth=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel (right side)
        control_frame = tk.Frame(main_frame, bg='#1a1a1a', width=300, relief=tk.RAISED, borderwidth=2)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        control_frame.pack_propagate(False)
        
        # Stats section
        stats_label = tk.Label(control_frame, text="📊 Statistics", 
                              font=('Arial', 14, 'bold'), bg='#1a1a1a', fg='white')
        stats_label.pack(pady=(10, 5))
        
        self.fps_label = tk.Label(control_frame, text="FPS: 0", 
                                 font=('Arial', 12), bg='#1a1a1a', fg='#00FF00')
        self.fps_label.pack(pady=5)
        
        self.detection_label = tk.Label(control_frame, text="Detections: 0", 
                                       font=('Arial', 12), bg='#1a1a1a', fg='#00FF00')
        self.detection_label.pack(pady=5)
        
        # Severity indicators
        tk.Label(control_frame, text="Severity Levels:", 
                font=('Arial', 12, 'bold'), bg='#1a1a1a', fg='white').pack(pady=(15, 5))
        
        for severity, color in self.severity_colors.items():
            frame = tk.Frame(control_frame, bg='#1a1a1a')
            frame.pack(pady=2)
            
            indicator = tk.Label(frame, text="●", font=('Arial', 16), 
                               bg='#1a1a1a', fg=color)
            indicator.pack(side=tk.LEFT, padx=5)
            
            tk.Label(frame, text=severity.capitalize(), font=('Arial', 11), 
                    bg='#1a1a1a', fg='white').pack(side=tk.LEFT)
        
        # Control buttons
        tk.Label(control_frame, text="Controls:", 
                font=('Arial', 12, 'bold'), bg='#1a1a1a', fg='white').pack(pady=(20, 10))
        
        self.start_btn = tk.Button(control_frame, text="▶ Start Detection", 
                                   command=self.start_detection,
                                   font=('Arial', 12, 'bold'),
                                   bg='#4CAF50', fg='white',
                                   activebackground='#45a049',
                                   relief=tk.RAISED, borderwidth=2,
                                   width=20, height=2)
        self.start_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(control_frame, text="■ Stop Detection", 
                                  command=self.stop_detection,
                                  font=('Arial', 12, 'bold'),
                                  bg='#f44336', fg='white',
                                  activebackground='#da190b',
                                  relief=tk.RAISED, borderwidth=2,
                                  width=20, height=2,
                                  state=tk.DISABLED)
        self.stop_btn.pack(pady=5)
        
        # Camera selection
        tk.Label(control_frame, text="Camera ID:", 
                font=('Arial', 10), bg='#1a1a1a', fg='white').pack(pady=(15, 0))
        
        self.camera_var = tk.StringVar(value="0")
        camera_entry = tk.Entry(control_frame, textvariable=self.camera_var,
                               font=('Arial', 11), width=10, justify='center')
        camera_entry.pack(pady=5)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#1a1a1a', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(status_frame, text="⚫ Ready", 
                                     font=('Arial', 10), bg='#1a1a1a', fg='#888')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
    def start_detection(self):
        """Start the detection process"""
        if self.running:
            return
        
        camera_id = int(self.camera_var.get())
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            self.status_label.config(text="❌ Error: Cannot open camera", fg='#FF0000')
            return
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="🔴 Detection Active", fg='#00FF00')
        
        # Start detection thread
        thread = threading.Thread(target=self.detection_loop, daemon=True)
        thread.start()
        
    def stop_detection(self):
        """Stop the detection process"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⚫ Stopped", fg='#888')
        
    def detection_loop(self):
        """Main detection loop"""
        fps_start = time.time()
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame, conf=0.45, verbose=False)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 30:
                self.fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
                frame_count = 0
            
            # Count detections
            self.detection_count = len(results[0].boxes)
            
            # Update display
            self.update_frame(annotated_frame)
            self.update_stats()
            
    def update_frame(self, frame):
        """Update the video frame in GUI"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit window
        height, width = frame_rgb.shape[:2]
        max_width = 900
        max_height = 680
        
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
    def update_stats(self):
        """Update statistics display"""
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        self.detection_label.config(text=f"Detections: {self.detection_count}")
        
    def run(self):
        """Start the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()

def main():
    import sys
    
    model_path = 'models/track_error_model/weights/best.pt'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print("🚂 Starting Track Error Detection GUI...")
    app = TrackDetectorGUI(model_path)
    app.run()

if __name__ == "__main__":
    main()

