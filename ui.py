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
    if "last_suggestion" not in st.session_state:
        st.session_state.last_suggestion = "Waiting for first analysis..."
    if "last_suggestion_time" not in st.session_state:
        st.session_state.last_suggestion_time = None

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

        # --- ROW 1: Monitoring Mode + Scene State ---
        row1_col1, row1_col2 = st.columns([1.5, 2.5], gap="large")

        with row1_col1:
            with st.container(border=True):
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

        with row1_col2:
            with st.container(border=True):
                st.subheader("Scene State")
                scene_state_placeholder = st.empty()
                scene_state_placeholder.caption("Scene state will appear here once monitoring starts.")

        st.divider()

        # --- ROW 2: AI Suggestion ---
        with st.container(border=True):
            st.subheader("AI Suggestion")
            suggestion_placeholder = st.empty()
            suggestion_placeholder.caption("AI suggestions will appear here once monitoring starts.")

        st.divider()

        # --- ROW 3: Camera Feed Toggle + Controls ---
        feed_col1, feed_col2 = st.columns([3, 1])
        with feed_col1:
            show_feed = st.toggle("📷 Show Live Camera Feed", value=False)
        with feed_col2:
            pass

        # Controls right below toggle
        control_col1, control_col2 = st.columns(2)
        with control_col1:
            start_btn = st.button("▶ Start Monitoring", use_container_width=True, type="primary")
        with control_col2:
            stop_btn = st.button("⏹ Stop Monitoring", use_container_width=True)

        if start_btn:
            st.session_state.running = True
        if stop_btn:
            st.session_state.running = False
            if st.session_state.camera_manager:
                st.session_state.camera_manager.stop_all()
                st.session_state.camera_manager = None
            # Clear all stream placeholders
            if "stream_placeholders" in st.session_state:
                for cam_id, ph in st.session_state.stream_placeholders.items():
                    ph["frame"].empty()
                del st.session_state.stream_placeholders
                

        # --- ROW 4: Camera Feed (conditional) ---
        if show_feed:
            with st.container(border=True):
                st.subheader("Live Camera Feed")
                feed_placeholder_area = st.empty()

        # Restore last suggestion on rerun
        if st.session_state.last_suggestion != "Waiting for first analysis...":
            suggestion_placeholder.success(
                f"**Mode:** `{category}`\n\n🕒 **{st.session_state.last_suggestion_time}**\n\n{st.session_state.last_suggestion}"
            )
        else:
            suggestion_placeholder.caption("AI suggestions will appear here once monitoring starts.")

        # --- Detection Loop ---
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

            # Create feed placeholders per camera only if feed is shown
            # Always recreate placeholders fresh on each rerun
            st.session_state.stream_placeholders = {}
            for cam_config in st.session_state.camera_configs:
                cam_id = cam_config["id"]
                if show_feed:
                    with st.container(border=True):
                        st.caption(f"📷 {cam_config['label']}")
                        frame_ph = st.empty()
                        frame_ph.info("📡 Connecting to camera feed...")
                else:
                    frame_ph = st.empty()
                    frame_ph.empty()
                st.session_state.stream_placeholders[cam_id] = {"frame": frame_ph}

            while st.session_state.running:
                streams = manager.get_all_streams()
            
                # Collect scene state from all cameras
                all_scene_data = []
                for cam_id, stream in streams.items():
                    if stream.error:
                        continue
                    
                    # Update feed if visible
                    if cam_id in st.session_state.stream_placeholders:
                        if show_feed:
                            frame = stream.get_frame()
                            if frame is not None:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                st.session_state.stream_placeholders[cam_id]["frame"].image(
                                    frame_rgb, use_column_width=True
                                )
                            else:
                                st.session_state.stream_placeholders[cam_id]["frame"].info("📡 Connecting...")
                        else:
                            # Feed toggled off — clear placeholder explicitly
                            st.session_state.stream_placeholders[cam_id]["frame"].empty()
            
                    # Collect scene data
                    scene = stream.get_scene()
                    if scene:
                        for obj, data in scene.items():
                            mins = data["duration_seconds"] // 60
                            count = data.get("count", 1)
                            all_scene_data.append({
                                "Camera": cam_id,
                                "Object": obj,
                                "Count": count,
                                "Status": data["status"],
                                "Duration (mins)": mins
                            })
            
                    # Update AI suggestion from first active stream
                    suggestion, s_time = stream.get_suggestion()
                    if suggestion and suggestion != "Waiting...":
                        st.session_state.last_suggestion = suggestion
                        st.session_state.last_suggestion_time = s_time
                        suggestion_placeholder.success(
                            f"**Mode:** `{category}`\n\n🕒 **{s_time}**\n\n{suggestion}"
                        )
            
                # Update scene state table
                if all_scene_data:
                    scene_state_placeholder.dataframe(
                        pd.DataFrame(all_scene_data),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    scene_state_placeholder.caption("No objects detected yet.")
            
                time.sleep(0.05)
            
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