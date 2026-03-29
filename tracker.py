from datetime import datetime
from storage import log_object

class ObjectTracker:
    def __init__(self, absence_threshold=10):
        # absence_threshold: seconds before object is marked absent
        self.tracked = {}
        self.absence_threshold = absence_threshold

    def update(self, detected_classes):
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # Update objects that are currently detected
        for obj in detected_classes:
            if obj not in self.tracked:
                # New object — start tracking it
                self.tracked[obj] = {
                    "first_seen": now_str,
                    "last_seen": now_str,
                    "status": "present",
                    "duration_seconds": 0
                }
            else:
                # Already tracked — update last seen and duration
                first = datetime.strptime(self.tracked[obj]["first_seen"], "%Y-%m-%d %H:%M:%S")
                self.tracked[obj]["last_seen"] = now_str
                self.tracked[obj]["duration_seconds"] = int((now - first).total_seconds())
                self.tracked[obj]["status"] = "present"

        # Mark objects not detected in this frame
        for obj in list(self.tracked.keys()):
            if obj not in detected_classes:
                last = datetime.strptime(self.tracked[obj]["last_seen"], "%Y-%m-%d %H:%M:%S")
                seconds_absent = int((now - last).total_seconds())

                if seconds_absent > self.absence_threshold:
                    if self.tracked[obj]["status"] == "present":
                        # Object just became absent — log it
                        log_object(
                            obj,
                            self.tracked[obj]["first_seen"],
                            self.tracked[obj]["last_seen"],
                            self.tracked[obj]["duration_seconds"],
                            "absent"
                        )
                    self.tracked[obj]["status"] = "absent"

    def get_scene_state(self):
        # Returns current state of all tracked objects
        return self.tracked

    def get_scene_description(self):
        # Builds a human-readable string for the LLM
        if not self.tracked:
            return "No objects detected on desk."

        parts = []
        for obj, data in self.tracked.items():
            duration = data["duration_seconds"]
            mins = duration // 60
            status = data["status"]

            if status == "present":
                parts.append(f"{obj} (present, {mins} mins)")
            else:
                parts.append(f"{obj} (absent, last seen {mins} mins ago)")

        return "Objects on desk: " + ", ".join(parts)