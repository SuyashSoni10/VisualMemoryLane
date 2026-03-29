import os
import time
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from storage import log_llm, log_action, log_summary
from plyer import notification
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """You are a smart desk assistant. Given the objects detected on a 
user's desk and how long they have been present or absent, suggest exactly one short, 
helpful, practical reminder or action for the user. Be concise, max 2 sentences."""

SUMMARY_PROMPT = """You are a workplace monitoring assistant. You will be given a list 
of objects and people detected at a desk over a 5 minute interval, including how long 
they were present or absent. Write a short, professional 2-3 sentence paragraph 
summarizing what happened during this interval. Focus on: how many people were present, 
how long they were at their desk, any notable absences, and any patterns worth flagging 
for a manager. Be factual and concise."""

LLM_INTERVAL = 60
SUMMARY_INTERVAL = 30  # 5 minutes
BOTTLE_ABSENCE_THRESHOLD = 1800

class ContextEngine:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.last_llm_call = 0
        self.last_summary_time = time.time()
        self.last_suggestion = "Waiting for first scene analysis..."
        self.last_suggestion_time = None

        # Track events within current 5 min interval
        self.interval_events = []
        self.interval_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run(self, tracker):
        now = time.time()

        # Check bottle absence
        self._check_bottle_alert(tracker)

        # Collect events for summary
        scene_state = tracker.get_scene_state()
        self._collect_interval_events(scene_state)

        # Call LLM every 60 seconds for suggestion
        if now - self.last_llm_call >= LLM_INTERVAL:
            scene_description = tracker.get_scene_description()
            suggestion = self._get_llm_suggestion(scene_description)
            self.last_suggestion = suggestion
            self.last_suggestion_time = datetime.now().strftime("%H:%M:%S")
            log_llm(scene_description, suggestion)
            self._notify("Desk Assistant", suggestion)
            self.last_llm_call = now

        # Generate 5 min summary
        if now - self.last_summary_time >= SUMMARY_INTERVAL:
            self._generate_interval_summary(tracker)
            self.last_summary_time = now

        return self.last_suggestion, self.last_suggestion_time

    def _collect_interval_events(self, scene_state):
        # Snapshot current scene state for summary context
        snapshot = []
        for obj, data in scene_state.items():
            mins = data["duration_seconds"] // 60
            snapshot.append(
                f"{obj} — {data['status']} for {mins} mins"
            )
        if snapshot:
            self.interval_events.append(", ".join(snapshot))

    def _generate_interval_summary(self, tracker):
        interval_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build context from collected events
        if not self.interval_events:
            event_context = "No objects detected during this interval."
        else:
            # Deduplicate and summarize events
            unique_events = list(dict.fromkeys(self.interval_events))
            event_context = f"During the interval from {self.interval_start} to {interval_end}, the following was observed: " + " | ".join(unique_events[-5:])

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": event_context}
                ]
            )
            summary = completion.choices[0].message.content
        except Exception as e:
            summary = f"Summary generation failed: {str(e)}"

        # Log to SQLite
        log_summary(self.interval_start, interval_end, summary)

        # Reset for next interval
        self.interval_events = []
        self.interval_start = interval_end

    def _get_llm_suggestion(self, scene_description):
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": scene_description}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"LLM error: {str(e)}"

    def _check_bottle_alert(self, tracker):
        state = tracker.get_scene_state()
        for obj, data in state.items():
            if "bottle" in obj.lower() and data["status"] == "absent":
                if data.get("duration_seconds", 0) >= BOTTLE_ABSENCE_THRESHOLD:
                    message = "You haven't had water in 30 minutes. Time to hydrate!"
                    self._notify("Hydration Reminder", message)
                    log_action("hydration_alert", message)

    def _notify(self, title, message):
        try:
            notification.notify(title=title, message=message, timeout=5)
        except Exception:
            pass