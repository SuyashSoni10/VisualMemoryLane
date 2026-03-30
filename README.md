# Visual Memory Lane

A camera-agnostic, context-aware visual AI assistant that passively observes your environment, logs objects over time, reasons about the scene using LLM, and triggers smart reminders.

---

## Architecture
```
Camera Input (Webcam / Phone / IP Cam)
        ↓
YOLOv8 — Object Detection
        ↓
Temporal Tracker — Object Persistence
        ↓
Groq LLaMA 3.1 — Scene Reasoning
        ↓
Action Layer — Notifications + Logging
        ↓
SQLite — Storage + Search
        ↓
Streamlit — Dashboard UI
```

---

## Features

- Real-time object detection via YOLOv8
- Tracks how long each object is present or absent
- LLM-powered smart suggestions every 60 seconds via Groq LLaMA 3.1
- Fully configurable absence alerts for any object — no hardcoding
- User category system — tailored AI reasoning per monitoring context
- AI-generated 5-minute interval summaries via Event History tab
- Desktop notifications for absence alerts and AI suggestions
- Searchable object history via SQLite
- Saves snapshot frames every 5 minutes
- Swappable camera source — webcam, Android phone, IP cam
- Sidebar configuration panel — no code changes needed

---

## User Categories

The system adapts its AI reasoning and language based on who is being monitored. Select a category from the dashboard before starting.

| Category | Focus |
|---|---|
| Student | Educational scaffolding, study habits, focus time |
| Patient (Alzheimer's) | Extreme patience, simple sentences, high empathy |
| Employee | Professional efficiency, actionable productivity items |
| Coach | Motivation, performance metrics, tough love encouragement |
| Teacher | Pedagogy, lesson flow, classroom management |
| Personal | Warm, informal, conversational daily life assistance |

Each category uses a custom system prompt for both real-time suggestions and 5-minute interval summaries.

---

## Absence Alerts

Absence alerts are fully configurable from the sidebar — no hardcoding required.

- Add any object to monitor (e.g. `bottle`, `person`, `medicine`, `phone`, `laptop`)
- Set a custom absence threshold in minutes per object
- Delete or modify rules at any time without restarting
- Alerts fire once per absence episode and reset when the object returns
- All triggered alerts are logged to SQLite with timestamp

---

## Event History & Interval Summaries

Every 5 minutes, the system automatically generates an AI-powered summary paragraph of everything observed at the desk during that interval. These summaries are stored in the SQLite database and accessible via the **Event History** tab in the dashboard.

### What the summary captures

- Which objects or people were present and for how long
- Notable absences during the interval
- How many times a previously absent subject returned
- Any patterns worth flagging
- Tone and focus adapted to the selected user category

### Example summary

> "During the 5-minute interval from 14:30 to 14:35, a person was present at the desk for approximately 3 minutes before leaving. The laptop remained active throughout the interval. The desk was unoccupied for the final 2 minutes, which may indicate a short break or distraction."

### Workplace monitoring use case

When deployed in an office environment via a fixed camera mount or existing CCTV, the system can passively track desk occupancy and generate per-interval reports. Managers can review the Event History tab to understand:

| What | How the system tracks it |
|---|---|
| Desk occupancy | Person detected / absent duration |
| Break patterns | Repeated absence in short intervals |
| Focus time | Continuous presence at desk |
| Return frequency | How often absent subjects returned |

### Accessing summaries

1. Run the app and click **Start**
2. Navigate to the **Event History** tab
3. Summaries appear automatically every 5 minutes
4. Click **Refresh History** to load the latest entries

> **Note:** For testing, set the summary interval to 30 seconds using the sidebar slider.

---

## Dashboard & Configuration

All configuration is managed from the sidebar — no code changes needed after setup.

### Sidebar options

- **Camera Source** — switch between laptop webcam, DroidCam USB, or IP camera stream
- **Absence Alert Rules** — add, edit, or delete object-specific absence thresholds
- **AI Suggestion Interval** — control how often the LLM generates a suggestion (30–300 seconds)
- **Summary Interval** — control how often interval summaries are generated (60–600 seconds)

### Dashboard tabs

- **Live Feed** — real-time camera feed with YOLO bounding boxes, scene state table, and AI suggestion box
- **Search** — search object history by name and view all logged events with timestamps
- **Event History** — 5-minute interval summaries with time range and AI-generated description

---

## Switch Camera Source

From the sidebar, select your camera input. No code changes needed.

For IP Camera, enter the stream URL in this format:
```
http://192.168.x.x:8080/video
```

To use an Android phone as a camera:
- Install **DroidCam** app for USB connection
- Install **IP Webcam** app for WiFi stream
- Both phone and laptop must be on the same WiFi network for IP Webcam
---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/visual-memory-lane.git
cd visual-memory-lane
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at https://console.groq.com

### 5. Run the app
```bash
streamlit run main.py
```

---

## Switch Camera Source

Open `detector.py` and change the `SOURCE` variable:
```python
SOURCE = 0                                 # Laptop webcam
SOURCE = 1                                 # DroidCam USB (Android)
SOURCE = "http://192.168.x.x:8080/video"  # IP Webcam stream
```

---

## Project Structure
```
visual-memory-lane/
├── main.py            # Entry point
├── detector.py        # YOLOv8 camera + detection
├── tracker.py         # Object persistence tracking
├── context_engine.py  # Groq LLM reasoning layer
├── storage.py         # SQLite logging + search
├── ui.py              # Streamlit dashboard
├── requirements.txt
├── README.md
├── .env               # API keys (never commit this)
└── frames/            # Auto-saved snapshots
```

---

## Future Roadmap

- CLIP-based visual search ("find my keys")
- Mobile app integration
- Smart glasses input
- Fine-tuned model for desk environment
- Multimodal LLM — send raw frames instead of text labels