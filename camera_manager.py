import threading
import cv2
import time
from datetime import datetime
from detector import Detector
from tracker import ObjectTracker
from context_engine import ContextEngine


FRAME_SAVE_INTERVAL = 300

class CameraStream:
    def __init__(self, camera_id, source, custom_classes, category,
                 alert_rules, llm_interval, summary_interval, voice_enabled):
        self.camera_id = camera_id
        self.source = source
        self.running = False
        self.current_frame = None
        self.current_scene = {}
        self.current_suggestion = "Waiting..."
        self.suggestion_time = None
        self.error = None
        self.lock = threading.Lock()
        self.detector = Detector(source=source, custom_classes=custom_classes)
        self.tracker = ObjectTracker()
        self.engine = ContextEngine(
            category=category,
            alert_rules=alert_rules,
            llm_interval=llm_interval,
            summary_interval=summary_interval,
            voice_enabled=voice_enabled,
            camera_id=camera_id
        )

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _run(self):
        try:
            self.detector.start()
        except RuntimeError as e:
            self.error = str(e)
            self.running = False
            return

        last_frame_save = time.time()

        while self.running:
            frame, detected_classes, detections_raw = self.detector.get_frame()

            if frame is None:
                self.error = "Camera feed lost"
                break

            self.tracker.update(detected_classes, detections_raw, frame, camera_id=self.camera_id)
            suggestion, suggestion_time = self.engine.run(self.tracker)

            if time.time() - last_frame_save >= FRAME_SAVE_INTERVAL:
                self._save_frame(frame)
                last_frame_save = time.time()

            with self.lock:
                self.current_frame = frame.copy()
                self.current_scene = self.tracker.get_scene_state()
                self.current_suggestion = suggestion
                self.suggestion_time = suggestion_time

            time.sleep(0.05)

        self.detector.stop()

    def _save_frame(self, frame):
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("frames", f"{self.camera_id}_{timestamp}.jpg")
        cv2.imwrite(path, frame)

    def get_frame(self):
        with self.lock:
            return self.current_frame

    def get_scene(self):
        with self.lock:
            return self.current_scene

    def get_suggestion(self):
        with self.lock:
            return self.current_suggestion, self.suggestion_time


class CameraManager:
    def __init__(self):
        self.streams = {}

    def add_camera(self, camera_id, source, custom_classes, category,
                   alert_rules, llm_interval, summary_interval, voice_enabled):
        if camera_id in self.streams:
            self.streams[camera_id].stop()

        stream = CameraStream(
            camera_id=camera_id,
            source=source,
            custom_classes=custom_classes,
            category=category,
            alert_rules=alert_rules,
            llm_interval=llm_interval,
            summary_interval=summary_interval,
            voice_enabled=voice_enabled
        )
        self.streams[camera_id] = stream
        stream.start()
        return stream

    def stop_all(self):
        for stream in self.streams.values():
            stream.stop()
        self.streams.clear()

    def get_all_streams(self):
        return self.streams