from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
from storage import log_object

class ObjectTracker:
    def __init__(self, absence_threshold=10):
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,           # frames to keep lost track alive
            n_init=3,             # frames before track is confirmed
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
        )
        self.absence_threshold = absence_threshold

        # Stores active tracks by ID
        # {track_id: {class, first_seen, last_seen, duration, status}}
        self.tracked = {}

        # Stores last known class per track ID
        self.id_to_class = {}

    def update(self, detected_classes, detections_raw, frame):
        """
        detected_classes: list of class name strings (for compatibility)
        detections_raw: list of dicts with keys: class, confidence, bbox [x1,y1,x2,y2]
        frame: current BGR frame from OpenCV (DeepSORT needs it for appearance features)
        """
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # Convert detections to DeepSORT format
        # DeepSORT expects: [([x1,y1,w,h], confidence, class_name), ...]
        ds_detections = []
        for det in detections_raw:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            ds_detections.append(
                ([x1, y1, w, h], det["confidence"], det["class"])
            )

        # Run DeepSORT update
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        # Track which IDs are active this frame
        active_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            class_name = track.det_class or self.id_to_class.get(track_id, "unknown")
            self.id_to_class[track_id] = class_name
            active_ids.add(track_id)

            if track_id not in self.tracked:
                # New confirmed track
                self.tracked[track_id] = {
                    "class": class_name,
                    "first_seen": now_str,
                    "last_seen": now_str,
                    "duration_seconds": 0,
                    "status": "present"
                }
            else:
                # Update existing track
                first = datetime.strptime(
                    self.tracked[track_id]["first_seen"],
                    "%Y-%m-%d %H:%M:%S"
                )
                self.tracked[track_id]["last_seen"] = now_str
                self.tracked[track_id]["duration_seconds"] = int(
                    (now - first).total_seconds()
                )
                self.tracked[track_id]["status"] = "present"

        # Mark absent tracks
        for track_id, data in list(self.tracked.items()):
            if track_id not in active_ids:
                last = datetime.strptime(data["last_seen"], "%Y-%m-%d %H:%M:%S")
                seconds_absent = int((now - last).total_seconds())

                if seconds_absent > self.absence_threshold:
                    if data["status"] == "present":
                        log_object(
                            data["class"],
                            data["first_seen"],
                            data["last_seen"],
                            data["duration_seconds"],
                            "absent"
                        )
                    self.tracked[track_id]["status"] = "absent"

    def get_scene_state(self):
        # Returns state keyed by class name for compatibility with context engine
        # If multiple IDs share a class, we merge them
        merged = {}
        for track_id, data in self.tracked.items():
            cls = data["class"]
            if cls not in merged:
                merged[cls] = {
                    "first_seen": data["first_seen"],
                    "last_seen": data["last_seen"],
                    "duration_seconds": data["duration_seconds"],
                    "status": data["status"],
                    "count": 1
                }
            else:
                # Multiple instances — take longest duration, present wins
                merged[cls]["duration_seconds"] = max(
                    merged[cls]["duration_seconds"],
                    data["duration_seconds"]
                )
                merged[cls]["count"] += 1
                if data["status"] == "present":
                    merged[cls]["status"] = "present"
        return merged

    def get_per_id_state(self):
        # Returns full per-ID tracking for detailed UI display
        return self.tracked

    def get_scene_description(self):
        state = get_scene_state_by_class(self)
        if not state:
            return "No objects detected."

        parts = []
        for cls, data in state.items():
            mins = data["duration_seconds"] // 60
            count = data.get("count", 1)
            count_str = f"{count}x " if count > 1 else ""
            status = data["status"]
            if status == "present":
                parts.append(f"{count_str}{cls} (present, {mins} mins)")
            else:
                parts.append(f"{count_str}{cls} (absent, last seen {mins} mins ago)")

        return "Objects on desk: " + ", ".join(parts)


def get_scene_state_by_class(tracker):
    return tracker.get_scene_state()