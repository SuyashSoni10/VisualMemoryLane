import cv2
from ultralytics import YOLO

# Change this to switch camera source:
# 0 = laptop webcam
# 1 = DroidCam USB (Android)
# "http://192.168.x.x:8080/video" = IP Webcam stream
SOURCE = 0

class Detector:
    def __init__(self, source=SOURCE):
        # Load the smallest YOLOv8 model — fast and accurate enough
        self.model = YOLO("yolov8n.pt")
        self.source = source
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
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