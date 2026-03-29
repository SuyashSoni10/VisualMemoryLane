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
- LLM-powered smart suggestions every 60 seconds
- Desktop notifications for hydration and activity reminders
- Searchable object history
- Saves snapshot frames every 5 minutes
- Swappable camera source — webcam, phone, IP cam

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
SOURCE = 0                              # Laptop webcam
SOURCE = 1                              # DroidCam USB (Android)
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
```