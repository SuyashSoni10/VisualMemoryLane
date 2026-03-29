import os
import time
from dotenv import load_dotenv
from groq import Groq
from storage import log_llm, log_action
from plyer import notification

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = "You are a smart desk assistant. Given the objects detected on a user's desk and how long they have been present or absent, suggest exactly one short, helpful, practical reminder or action for the user. Be concise, max 2 sentences."

# How often to call the LLM (seconds)
LLM_INTERVAL = 60

# How long bottle must be absent to trigger alert (seconds)
BOTTLE_ABSENCE_THRESHOLD = 1800  # 30 minutes

class ContextEngine:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.last_llm_call = 0
        self.last_suggestion = "Waiting for first scene analysis..."
        self.last_suggestion_time = None

    def run(self, tracker):
        now = time.time()

        # Check bottle absence alert
        self._check_bottle_alert(tracker)

        # Call LLM every 60 seconds
        if now - self.last_llm_call >= LLM_INTERVAL:
            scene_description = tracker.get_scene_description()
            suggestion = self._get_llm_suggestion(scene_description)
            self.last_suggestion = suggestion
            self.last_suggestion_time = time.strftime("%H:%M:%S")

            # Log to SQLite
            log_llm(scene_description, suggestion)

            # Send desktop notification
            self._notify("Desk Assistant", suggestion)

            self.last_llm_call = now

        return self.last_suggestion, self.last_suggestion_time

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

        # Check if bottle is tracked and absent too long
        for obj, data in state.items():
            if "bottle" in obj.lower() and data["status"] == "absent":
                absent_seconds = data.get("duration_seconds", 0)
                if absent_seconds >= BOTTLE_ABSENCE_THRESHOLD:
                    message = "You haven't had water in 30 minutes. Time to hydrate!"
                    self._notify("Hydration Reminder", message)
                    log_action("hydration_alert", message)

    def _notify(self, title, message):
        try:
            notification.notify(
                title=title,
                message=message,
                timeout=5
            )
        except Exception:
            # Notifications may not work on all systems — fail silently
            pass