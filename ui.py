import streamlit as st
import cv2
import time
import os
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd
import logging
import warnings

from detector import Detector
from tracker import ObjectTracker
from context_engine import ContextEngine
from storage import init_db, search_objects, get_recent_logs, get_latest_llm, get_summaries, log_object
from camera_manager import CameraManager
from clip_search import embed_frame

logging.getLogger("ultralytics").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

init_db()

FRAME_SAVE_INTERVAL = 300



def save_frame(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("frames", f"snapshot_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    # Immediately embed this frame for CLIP search
    embed_frame(path)

def main():
    st.set_page_config(page_title="Visual Memory Lane", layout="wide")

    st.title("Visual Memory Lane")
    st.markdown("A **context-aware visual AI assistant** that observes objects, tracks activity, and provides intelligent suggestions.")
    st.divider()

    # Shared state
    if "running" not in st.session_state:
        st.session_state.running = False
    if "suggestion" not in st.session_state:
        st.session_state.suggestion = "Waiting for first analysis..."
    if "suggestion_time" not in st.session_state:
        st.session_state.suggestion_time = None
    if "alert_rules" not in st.session_state:
        st.session_state.alert_rules = [{"object": "bottle", "minutes": 30}]
    if "camera_configs" not in st.session_state:
        st.session_state.camera_configs = [
            {"id": "cam_0", "source": "0", "label": "Webcam"}
        ]
    if "camera_manager" not in st.session_state:
        st.session_state.camera_manager = None

    # --- SIDEBAR ---
    source = 0  # default

    with st.sidebar:
        st.header("Configuration")
        #camera configurations setting block
        st.subheader("Camera Sources")
        st.caption("Add up to 4 camera sources.")

        updated_cameras = []
        for i, cam in enumerate(st.session_state.camera_configs):
            with st.expander(f"📷 {cam['label']}", expanded=i == 0):
                label = st.text_input("Label", value=cam["label"], key=f"cam_label_{i}")
                source_type = st.radio(
                    "Source type",
                    ["Webcam", "DroidCam USB", "IP Camera"],
                    key=f"cam_source_type_{i}"
                )
                if source_type == "Webcam":
                    source = "0"
                elif source_type == "DroidCam USB":
                    source = "1"
                else:
                    source = st.text_input(
                        "IP stream URL",
                        value="http://192.168.x.x:8080/video",
                        key=f"cam_ip_{i}"
                    )
                delete_cam = st.button("Remove", key=f"del_cam_{i}")
                if not delete_cam:
                    updated_cameras.append({
                        "id": f"cam_{i}",
                        "source": source,
                        "label": label
                    })

        st.session_state.camera_configs = updated_cameras

        if len(st.session_state.camera_configs) < 4:
            if st.button("+ Add Camera"):
                st.session_state.camera_configs.append({
                    "id": f"cam_{len(st.session_state.camera_configs)}",
                    "source": "0",
                    "label": f"Camera {len(st.session_state.camera_configs) + 1}"
                })
                st.rerun()
        st.divider()
        st.subheader("Detection Classes")
        st.caption("What objects to detect. Separate with commas.")
        classes_input = st.text_area(
            "Objects",
            value="person, bottle, laptop, phone, cup, book, chair, keyboard, mouse",
            height=100
        )
        custom_classes = [c.strip() for c in classes_input.split(",") if c.strip()]
        
        st.divider()
        st.subheader("Voice Alerts")
        voice_enabled = st.toggle("Enable Voice alerts", value = False)
        
        st.divider()

        st.subheader("Absence Alerts")
        st.caption("Get notified when any object is absent too long.")

        updated_rules = []
        for i, rule in enumerate(st.session_state.alert_rules):
            col_obj, col_min, col_del = st.columns([3, 2, 1])
            with col_obj:
                obj = st.text_input("Object", value=rule["object"], key=f"obj_{i}")
            with col_min:
                mins = st.number_input("Minutes", min_value=1, max_value=480, value=rule["minutes"], key=f"min_{i}")
            with col_del:
                st.write("")
                st.write("")
                delete = st.button("✕", key=f"del_{i}")
            if not delete:
                updated_rules.append({"object": obj, "minutes": mins})

        st.session_state.alert_rules = updated_rules

        if st.button("+ Add Alert Rule"):
            st.session_state.alert_rules.append({"object": "", "minutes": 30})
            st.rerun()

        st.divider()

        st.subheader("Intervals")
        llm_interval = st.slider(
            "AI suggestion every (seconds)",
            min_value=30, max_value=300, value=60, step=10
        )
        summary_interval = st.slider(
            "Summary every (seconds)",
            min_value=60, max_value=600, value=300, step=60
        )

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Live Feed", "Search", "Event History", "Chat"])

    # --- TAB 1: Live Feed ---
    with tab1:
        col1, col2 = st.columns([2.5, 1.5], gap="large")

        with col1:
            with st.container(border=True):
                st.subheader("Live Camera Feed")
                frame_placeholder = st.empty()

        with col2:
            with st.container(border=True):
                st.subheader("Scene State")
                scene_placeholder = st.empty()

            st.subheader("Monitoring Mode")
            category = st.selectbox(
                "Select user profile",
                options=[
                    "Student",
                    "Patient (Alzheimer's)",
                    "Employee",
                    "Coach",
                    "Teacher",
                    "Personal"
                ],
                index=5
            )

            category_descriptions = {
                "Student": "📚 Educational scaffolding and study habit reminders.",
                "Patient (Alzheimer's)": "🤍 Gentle, simple, empathetic guidance.",
                "Employee": "💼 Professional efficiency and productivity focus.",
                "Coach": "🏆 Motivation, performance metrics, tough love.",
                "Teacher": "🎓 Pedagogy, lesson flow, classroom management.",
                "Personal": "😊 Warm, casual, daily life assistance."
            }
            st.info(category_descriptions[category])

        st.divider()

        with st.container(border=True):
            st.subheader("AI Suggestion")
            suggestion_placeholder = st.empty()

        st.subheader("Controls")
        control_col1, control_col2 = st.columns(2)
        with control_col1:
            start_btn = st.button("▶ Start Monitoring", use_container_width=True, type="primary")
        with control_col2:
            stop_btn = st.button("⏹ Stop Monitoring", use_container_width=True)

        if start_btn:
            st.session_state.running = True
        if stop_btn:
            st.session_state.running = False

        # Detection Loop 
        if st.session_state.running:
            

            if st.session_state.camera_manager is None:
                st.session_state.camera_manager = CameraManager()

            manager = st.session_state.camera_manager

            for cam_config in st.session_state.camera_configs:
                src = int(cam_config["source"]) if cam_config["source"].isdigit() else cam_config["source"]
                manager.add_camera(
                    camera_id=cam_config["id"],
                    source=src,
                    custom_classes=custom_classes,
                    category=category,
                    alert_rules=st.session_state.alert_rules,
                    llm_interval=llm_interval,
                    summary_interval=summary_interval,
                    voice_enabled=voice_enabled
                )

            # Create placeholders OUTSIDE the loop
            stream_placeholders = {}
            for cam_config in st.session_state.camera_configs:
                cam_id = cam_config["id"]
                st.markdown(f"**📷 {cam_config['label']}**")
                col_feed, col_scene = st.columns([2, 1])
                with col_feed:
                    frame_ph = st.empty()
                with col_scene:
                    scene_ph = st.empty()
                suggestion_ph = st.empty()
                stream_placeholders[cam_id] = {
                    "frame": frame_ph,
                    "scene": scene_ph,
                    "suggestion": suggestion_ph
                }

            # Now loop only updates placeholders
            while st.session_state.running:
                streams = manager.get_all_streams()

                for cam_id, stream in streams.items():
                    if cam_id not in stream_placeholders:
                        continue
                    
                    ph = stream_placeholders[cam_id]

                    if stream.error:
                        ph["frame"].error(f"Camera error: {stream.error}")
                        continue
                    
                    frame = stream.get_frame()
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ph["frame"].image(frame_rgb, use_column_width=True)

                    scene = stream.get_scene()
                    if scene:
                        table_data = []
                        for obj, data in scene.items():
                            mins = data["duration_seconds"] // 60
                            count = data.get("count", 1)
                            table_data.append({
                                "Object": obj,
                                "Count": count,
                                "Status": data["status"],
                                "Mins": mins
                            })
                        ph["scene"].dataframe(
                            pd.DataFrame(table_data),
                            use_container_width=True,
                            hide_index=True
                        )

                    suggestion, s_time = stream.get_suggestion()
                    if suggestion:
                        ph["suggestion"].success(
                            f"**Mode:** `{category}`\n\n🕒 **{s_time}**\n\n{suggestion}"
                        )

                time.sleep(0.05)

        if stop_btn:
            if st.session_state.camera_manager:
                st.session_state.camera_manager.stop_all()
                st.session_state.camera_manager = None

    # --- TAB 2: Search ---
    with tab2:
        st.subheader("Search Object History")
        query = st.text_input(
            "Search object history",
            placeholder="Example: bottle, laptop, book..."
        )
        
        if query:
            results = search_objects(query)
            if results:
                st.write(f"Found {len(results)} result(s) for **{query}**:")
                for r in results:
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{r[0]}**")
                            st.caption(f"First seen: {r[1]}  •  Last seen: {r[2]}")
                        with col2:
                            duration = r[3] if r[3] is not None else 0
                            st.metric("Duration", f"{duration // 60} mins")
                            status_color = "🟢" if r[4] == "present" else "🔴"
                            st.write(f"{status_color} {r[4]}")
            else:
                st.warning(f"No results found for '{query}'")

    # --- TAB 3: Event History ---
    with tab3:
        st.subheader("5-Minute Interval Summaries")
        st.caption("AI-generated summary of what was observed at the desk every 5 minutes.")

        summaries = get_summaries(20)

        if summaries:
            for s in summaries:
                with st.container(border=True):
                    col_time, col_summary = st.columns([2, 5])
                    with col_time:
                        st.markdown(f"**{s[0]}**")
                        st.caption(f"→ {s[1]}")
                    with col_summary:
                        st.write(s[2])
        else:
            st.info("No summaries yet. Summaries are generated every 5 minutes while the feed is running.")

        if st.button("Refresh History"):
            st.rerun()
    
    # --- TAB 4: Chat ---
    with tab4:
        st.subheader("💬 Ask your workspace assistant")
        st.caption("Ask anything about what has been observed in your workspace.")

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask something... e.g. 'What was on my desk this morning?'")

        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            # Call FastAPI chat endpoint
            import requests
            try:
                res = requests.post(
                    "http://localhost:8000/chat/message",
                    json={
                        "message": user_input,
                        "category": category if "category" in dir() else "Personal"
                    },
                    timeout=15
                )
                response = res.json().get("response", "No response received.")
            except Exception as e:
                response = f"Could not reach assistant: {str(e)}"

            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

            st.rerun()

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("Clear chat"):
                st.session_state.chat_history = []
                st.rerun()
if __name__ == "__main__":
    main()