# Visual Memory Lane

> A camera-agnostic, context-aware visual AI assistant that passively observes your environment, tracks objects over time, reasons about scenes using LLM, and assists users through a conversational interface.

---

## What is this?

Visual Memory Lane is a multi-camera AI system that watches your workspace and acts as a second pair of eyes. It detects objects, tracks how long they've been present or absent, generates intelligent suggestions, and answers natural language questions about what it has observed.

Originally conceived as an assistive tool for Alzheimer's patients, students, engineers, and workplace monitoring — it adapts its intelligence based on who is being monitored.

---

## Architecture

```
Camera Inputs (Webcam / Phone / IP Cam / Smart Glasses)
        ↓
YOLO11 — Real-time Object Detection
        ↓
DeepSORT — Per-ID Object Tracking
        ↓
Temporal Tracker — Object Persistence Over Time
        ↓
Groq LLaMA 3.1 — Scene Reasoning + Conversational AI
        ↓
Action Layer — Voice Alerts + Desktop Notifications + Logging
        ↓
PostgreSQL — Persistent Storage + Search
        ↓
FastAPI Backend — REST API for all services
        ↓
Streamlit Dashboard + React Native Mobile App
```

---

## Features

- Real-time object detection via YOLO11
- Per-ID object tracking via DeepSORT — distinguishes multiple instances of the same class
- Multi-camera support — up to 4 simultaneous camera feeds
- LLM-powered smart suggestions every 60 seconds via Groq LLaMA 3.1
- Conversational chat interface — ask questions about your workspace in natural language
- AI-generated 5-minute interval summaries stored in PostgreSQL
- User category system — 6 profiles with tailored AI reasoning
- Fully configurable absence alerts for any object
- Voice alerts via pyttsx3
- Desktop notifications via plyer
- CLIP-based visual search — search saved frames using natural language
- Searchable object history via PostgreSQL
- Snapshot frames saved every 5 minutes
- JWT authentication — multi-user support
- React Native mobile app — login, search, summaries
- Docker support for deployment
- Camera-agnostic — webcam, Android phone, IP camera, smart glasses ready

---

## User Categories

The system adapts its AI reasoning and language based on who is being monitored.

| Category | Focus |
|---|---|
| Student | Educational scaffolding, study habits, focus time |
| Patient (Alzheimer's) | Extreme patience, simple sentences, high empathy |
| Employee | Professional efficiency, actionable productivity items |
| Coach | Motivation, performance metrics, tough love encouragement |
| Teacher | Pedagogy, lesson flow, classroom management |
| Personal | Warm, informal, conversational daily life assistance |

---

## Event History & Interval Summaries

Every 5 minutes the system generates an AI-powered summary of everything observed. Summaries are stored in PostgreSQL and accessible via the Event History tab.

### What the summary captures

- Which objects or people were present and for how long
- Notable absences during the interval
- How many times a previously absent subject returned
- Any patterns worth flagging
- Tone and focus adapted to the selected user category

### Example summary

> "During the 5-minute interval from 14:30 to 14:35, a person was present at the desk for approximately 3 minutes before leaving. The laptop remained active throughout the interval. The desk was unoccupied for the final 2 minutes, which may indicate a short break or distraction."

### Workplace monitoring use case

When deployed in an office via fixed camera mounts or existing CCTV, the system tracks desk occupancy and generates per-interval reports.

| What | How the system tracks it |
|---|---|
| Desk occupancy | Person detected / absent duration |
| Break patterns | Repeated absence in short intervals |
| Focus time | Continuous presence at desk |
| Return frequency | How often absent subjects returned |

---

## Conversational Interface

Ask natural language questions about your workspace directly in the Chat tab.

**Example queries:**
- "What objects have been on my desk today?"
- "Was anyone at the desk this morning?"
- "What was the last AI suggestion?"
- "How long has my laptop been present?"

The assistant reasons over your full PostgreSQL history to answer accurately.

---

## Absence Alerts

Fully configurable from the sidebar — no code changes needed.

- Add any object to monitor: `bottle`, `person`, `medicine`, `phone`, `laptop`
- Set a custom absence threshold in minutes per object
- Alerts fire once per absence episode and reset when the object returns
- Desktop notification + voice alert + logged to PostgreSQL

---

## Multi-Camera Support

Run up to 4 simultaneous camera feeds from the sidebar. Each camera runs in its own thread with independent detection, tracking, and context engine. All feeds write to shared PostgreSQL storage tagged by `camera_id`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Detection | YOLO11 (Ultralytics) |
| Tracking | DeepSORT |
| Visual search | CLIP (OpenAI) |
| LLM reasoning | Groq LLaMA 3.1 8B Instant |
| Backend API | FastAPI + Uvicorn |
| Database | PostgreSQL via SQLAlchemy |
| Auth | JWT via python-jose + passlib |
| Desktop UI | Streamlit |
| Mobile app | React Native (Expo) |
| Voice | pyttsx3 |
| Notifications | plyer |
| Deployment | Docker + docker-compose |
| Language | Python 3.10+ / JavaScript |

---

## Project Structure

```
visual-memory-lane/
├── main.py                  # Entry point
├── detector.py              # YOLO11 camera + detection
├── tracker.py               # DeepSORT object tracking
├── context_engine.py        # Groq LLM reasoning layer
├── camera_manager.py        # Multi-camera thread manager
├── clip_search.py           # CLIP visual search
├── voice.py                 # pyttsx3 voice alerts
├── storage.py               # PostgreSQL logging + search
├── ui.py                    # Streamlit dashboard
├── api/
│   ├── main.py              # FastAPI app
│   ├── routes/
│   │   ├── auth.py          # JWT register + login
│   │   ├── search.py        # Search endpoints
│   │   ├── chat.py          # Conversational AI endpoint
│   │   └── notifications.py # Push notification endpoints
│   ├── models/
│   │   ├── database.py      # SQLAlchemy models
│   │   └── schemas.py       # Pydantic schemas
│   └── core/
│       └── security.py      # JWT + bcrypt security
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                     # API keys (never commit this)
├── .gitignore
└── frames/                  # Auto-saved snapshots

vml-mobile/                  # React Native mobile app
├── App.js
├── screens/
│   ├── LoginScreen.js
│   ├── HomeScreen.js
│   ├── SearchScreen.js
│   └── SummariesScreen.js
└── services/
    ├── api.js
    └── notifications.js
```

---

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Node.js 18+ (for mobile app)
- Expo Go app on your phone

### 1. Clone the repo

```bash
git clone https://github.com/SuyashSoni10/VisualMemoryLane.git
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

### 4. Set up PostgreSQL

```sql
CREATE DATABASE visual_memory_lane;
CREATE USER vml_user WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE visual_memory_lane TO vml_user;
GRANT ALL ON SCHEMA public TO vml_user;
```

### 5. Configure environment variables

Create a `.env` file in the root folder:

```
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://vml_user:yourpassword@localhost:5432/visual_memory_lane
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

Get your free Groq API key at https://console.groq.com

Generate a secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 6. Run the backend API

```bash
uvicorn api.main:app --reload --port 8000 --host 0.0.0.0
```

### 7. Run the desktop app

```bash
streamlit run main.py
```

### 8. Run with Docker

```bash
docker-compose up --build
```

---

## Switch Camera Source

From the sidebar, add camera sources dynamically. Supported inputs:

```
Webcam          → source: 0
DroidCam USB    → source: 1
IP Webcam       → http://192.168.x.x:8080/video
```

For Android phone as camera:
- USB: Install DroidCam app
- WiFi: Install IP Webcam app (both devices on same network)

---

## Mobile App Setup

```bash
cd vml-mobile
npm install
```

Update `BASE_URL` in `services/api.js` with your laptop's IP:
```javascript
const BASE_URL = 'http://YOUR_LAPTOP_IP:8000';
```

Run:
```bash
npx expo start
```

Scan QR code with Expo Go on your phone.

---

## API Documentation

Once the backend is running, visit:
```
http://localhost:8000/docs
```

Full Swagger UI with all endpoints.

---

## Dashboard Tabs

| Tab | What it shows |
|---|---|
| Live Feed | Real-time camera feeds with bounding boxes, scene state table, AI suggestion |
| Search | Search object history by name across all cameras |
| Event History | 5-minute AI-generated interval summaries |
| Chat | Conversational AI — ask anything about your workspace |

---

## Sidebar Configuration

All configuration managed from sidebar — no code changes needed.

| Setting | Description |
|---|---|
| Camera Sources | Add up to 4 cameras, label them, switch source type |
| Detection Classes | Define what objects to detect |
| Voice Alerts | Toggle voice on/off |
| Absence Alert Rules | Add any object with custom threshold in minutes |
| AI Suggestion Interval | 30–300 seconds |
| Summary Interval | 60–600 seconds |
| User Category | Switch monitoring profile |

---

## Future Roadmap

- YOLO-World open vocabulary detection — detect any custom object by text description
- Fine-tuned model on desk/workspace data
- Multimodal LLM — send raw frames instead of text labels
- Smart glasses integration
- Push notifications via Firebase
- React web dashboard replacing Streamlit
- Real-time multi-user collaboration
- Privacy controls — blur faces, local-only mode

---

## License

MIT License — free to use, modify, and distribute.