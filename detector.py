import streamlit as st
import cv2
from ultralytics import YOLO
import logging

# Change this to switch camera source:
# 0 = laptop webcam
# 1 = DroidCam USB (Android)
# "http://192.168.x.x:8080/video" = IP Webcam stream
#SOURCE = "http://172.25.235.218:8080/video"
SOURCE = 0
class Detector:
    def __init__(self, source=None):
        # Load the smallest YOLOv8 model — fast and accurate enough
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        self.model = YOLO("yolov8n.pt", verbose=False)
        self.source = source if source is not None else SOURCE
        self.cap = None
    
    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        
        # Set timeout for IP streams so it fails fast instead of hanging
        if isinstance(self.source, str):
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.source}")
    
    def stop(self):
        if self.cap:
            self.cap.release()

    def get_frame(self):
        # Read one frame from camera
        if not self.cap:
            raise RuntimeError("Camera not started. Call start() first.")
        ret, frame = self.cap.read()
        if not ret:
            return None, [], None
        return self._process_frame(frame)

    def _process_frame(self, frame):
        # Run YOLO detection on the frame
        results = self.model(frame, verbose=False)

        # Get annotated frame with bounding boxes drawn
        annotated_frame = results[0].plot()

        # Extract unique detected class names
        detected_classes = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            if class_name not in detected_classes:
                detected_classes.append(class_name)

        # Get confidence scores for display
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2)
            })

        return annotated_frame, detected_classes, detections