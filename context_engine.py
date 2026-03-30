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

# System prompts per user category
CATEGORY_PROMPTS = {
    "Student": {
        "suggestion": """You are a smart study assistant monitoring a student's desk. 
Given the objects detected and their durations, provide one clear, encouraging, 
educational reminder. Focus on study habits, focus time, and learning productivity. 
Use simple, motivating language. Max 2 sentences.""",

        "summary": """You are an educational monitoring assistant. Given what was 
observed at a student's desk over the last 5 minutes, write a short 2-3 sentence 
professional summary. Focus on: study session duration, distractions, materials 
present, and focus quality. Use clear and constructive language suitable for 
academic progress tracking."""
    },

    "Patient (Alzheimer's)": {
        "suggestion": """You are a gentle, caring assistant helping someone with 
Alzheimer's disease. Given what is detected on their desk or in their space, 
provide one extremely simple, warm, and reassuring reminder. Use very short 
sentences. Be calm, repetitive if needed, and never alarming. Max 2 sentences.""",

        "summary": """You are a patient care monitoring assistant. Given what was 
observed in the patient's space over the last 5 minutes, write a short 2-3 sentence 
summary in simple language. Focus on: presence and movement, any concerning absences, 
medication or water bottle status, and general activity level. Use empathetic and 
clinical language suitable for a caregiver report."""
    },

    "Employee": {
        "suggestion": """You are a professional workplace assistant monitoring an 
employee's desk. Given the detected objects and durations, provide one concise, 
actionable productivity reminder. Focus on task focus, desk organization, and 
professional efficiency. Be direct and professional. Max 2 sentences.""",

        "summary": """You are a workplace productivity monitoring assistant. Given 
what was observed at an employee's desk over the last 5 minutes, write a short 
2-3 sentence professional summary. Focus on: desk occupancy, focus time, absence 
patterns, and productivity indicators. Use formal, corporate language suitable 
for a manager's report."""
    },

    "Coach": {
        "suggestion": """You are a high-performance coach monitoring an athlete or 
coachee's workspace. Given the detected objects and durations, deliver one sharp, 
motivating, performance-focused reminder. Be bold, direct, and energizing. 
Push for excellence. Max 2 sentences.""",

        "summary": """You are a performance coaching assistant. Given what was 
observed at the coachee's desk or training area over the last 5 minutes, write 
a short 2-3 sentence summary. Focus on: engagement level, consistency, any 
drop in presence or activity, and performance patterns. Use motivational, 
results-driven language."""
    },

    "Teacher": {
        "suggestion": """You are a pedagogical assistant monitoring a teacher's 
workspace or classroom. Given the detected objects and durations, provide one 
practical, classroom-focused reminder. Focus on lesson flow, materials management, 
and student engagement indicators. Be professional and supportive. Max 2 sentences.""",

        "summary": """You are a classroom monitoring assistant. Given what was 
observed in the teacher's space over the last 5 minutes, write a short 2-3 sentence 
summary. Focus on: teacher presence, classroom materials activity, any disruptions, 
and lesson flow indicators. Use professional educational language."""
    },

    "Personal": {
        "suggestion": """You are a warm, friendly personal assistant monitoring 
someone's home desk or living space. Given the detected objects and durations, 
give one casual, helpful, friendly reminder suited for daily life. Be conversational, 
light, and natural. Max 2 sentences.""",

        "summary": """You are a personal life assistant. Given what was observed 
in someone's personal space over the last 5 minutes, write a short 2-3 sentence 
friendly summary. Focus on: activity patterns, wellbeing indicators like water 
and food, rest vs work balance, and any notable events. Use a warm, informal tone."""
    }
}

LLM_INTERVAL = 60
SUMMARY_INTERVAL = 300
BOTTLE_ABSENCE_THRESHOLD = 1800


class ContextEngine:
    def __init__(self, category="Personal"):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.category = category
        self.last_llm_call = 0
        self.last_summary_time = time.time()
        self.last_suggestion = "Waiting for first scene analysis..."
        self.last_suggestion_time = None
        self.interval_events = []
        self.interval_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def set_category(self, category):
        # Allow category to be changed at runtime from UI
        self.category = category

    def _get_prompts(self):
        return CATEGORY_PROMPTS.get(self.category, CATEGORY_PROMPTS["Personal"])

    def run(self, tracker):
        now = time.time()

        self._check_bottle_alert(tracker)
        self._collect_interval_events(tracker.get_scene_state())

        if now - self.last_llm_call >= LLM_INTERVAL:
            scene_description = tracker.get_scene_description()
            suggestion = self._get_llm_suggestion(scene_description)
            self.last_suggestion = suggestion
            self.last_suggestion_time = datetime.now().strftime("%H:%M:%S")
            log_llm(scene_description, suggestion)
            self._notify("Desk Assistant", suggestion)
            self.last_llm_call = now

        if now - self.last_summary_time >= SUMMARY_INTERVAL:
            self._generate_interval_summary(tracker)
            self.last_summary_time = now

        return self.last_suggestion, self.last_suggestion_time

    def _get_llm_suggestion(self, scene_description):
        try:
            prompt = self._get_prompts()["suggestion"]
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": scene_description}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"LLM error: {str(e)}"

    def _collect_interval_events(self, scene_state):
        snapshot = []
        for obj, data in scene_state.items():
            mins = data["duration_seconds"] // 60
            snapshot.append(f"{obj} — {data['status']} for {mins} mins")
        if snapshot:
            self.interval_events.append(", ".join(snapshot))

    def _generate_interval_summary(self, tracker):
        interval_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not self.interval_events:
            event_context = "No objects detected during this interval."
        else:
            unique_events = list(dict.fromkeys(self.interval_events))
            event_context = (
                f"User category: {self.category}. "
                f"During the interval from {self.interval_start} to {interval_end}, "
                f"the following was observed: " + " | ".join(unique_events[-5:])
            )

        try:
            prompt = self._get_prompts()["summary"]
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": event_context}
                ]
            )
            summary = completion.choices[0].message.content
        except Exception as e:
            summary = f"Summary generation failed: {str(e)}"

        log_summary(self.interval_start, interval_end, summary)
        self.interval_events = []
        self.interval_start = interval_end

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