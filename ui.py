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

    # --- SIDEBAR ---
    source = 0  # default

    with st.sidebar:
        st.header("Configuration")

        st.subheader("Camera Source")
        camera_source = st.radio(
            "Select input",
            options=["Laptop Webcam", "DroidCam USB", "IP Camera"]
        )
        if camera_source == "Laptop Webcam":
            source = 0
        elif camera_source == "DroidCam USB":
            source = 1
        else:
            source = st.text_input(
                "Enter IP stream URL",
                "http://192.168.x.x:8080/video"
            )

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
    tab1, tab2, tab3, tab4 = st.tabs(["Live Feed", "Search", "Event History", "Visual Search"])

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

        if st.session_state.running:
            detector = Detector(source=source)
            tracker = ObjectTracker()
            engine = ContextEngine(
                category=category,
                alert_rules=st.session_state.alert_rules,
                llm_interval=llm_interval,
                summary_interval=summary_interval
            )

            try:
                detector.start()
            except RuntimeError as e:
                st.error(f"Camera error: {str(e)}. Check your camera source in the sidebar.")
                st.session_state.running = False
                st.stop()

            last_frame_save = time.time()

            while st.session_state.running:
                frame, detected_classes, detections_raw = detector.get_frame()

                if frame is None:
                    st.error("Camera feed lost.")
                    break

                tracker.update(detected_classes, detections_raw, frame)
                suggestion, suggestion_time = engine.run(tracker)

                if time.time() - last_frame_save >= FRAME_SAVE_INTERVAL:
                    save_frame(frame)
                    last_frame_save = time.time()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, use_column_width=True)

                scene_state = tracker.get_scene_state()
                
                if scene_state:
                    table_data = []
                    for obj, data in scene_state.items():
                        mins = data["duration_seconds"] // 60
                        count = data.get("count", 1)
                        table_data.append({
                            "Object": obj,
                            "Count": count,
                            "Status": data["status"],
                            "Duration (mins)": mins,
                            "First Seen": data["first_seen"]
                        })
                        scene_placeholder.dataframe(
                            pd.DataFrame(table_data),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    scene_placeholder.info("No objects detected yet.")

                if suggestion:
                    suggestion_placeholder.success(
                        f"""
**Mode:** `{category}`  

🕒 **{suggestion_time}**

{suggestion}
"""
                    )

                time.sleep(0.05)

            detector.stop()

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
                st.write(f"Found {len(results)} result(s):")
                for r in results:
                    st.write(f"**{r[0]}** | ⏱ Duration: {r[3]//60} mins | Status: `{r[4]}`")
                    st.caption(f"First seen: {r[1]}  •  Last seen: {r[2]}")
            else:
                st.warning("No results found.")

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
    
    # --- TAB 4: Visual Search ---
    with tab4:
        st.subheader("Visual Search")
        st.caption("Search your saved snapshots using natural language.")
    
        col_search, col_btn = st.columns([4, 1])
        with col_search:
            visual_query = st.text_input(
                "Describe what you're looking for",
                placeholder="e.g. empty desk, person using laptop, water bottle..."
            )
        with col_btn:
            st.write("")
            embed_btn = st.button("Index frames", help="Re-index all saved frames")
    
        if embed_btn:
            from clip_search import embed_all_frames
            with st.spinner("Indexing all frames..."):
                embed_all_frames()
            st.success("All frames indexed.")
    
        if visual_query:
            from clip_search import search_frames
            results = search_frames(visual_query, top_k=5)
    
            if results:
                st.write(f"Top {len(results)} matches:")
                cols = st.columns(len(results))
                for i, (frame_path, score) in enumerate(results):
                    with cols[i]:
                        try:
                            st.image(frame_path, use_column_width=True)
                            st.caption(f"Score: {score:.2f}")
                            st.caption(frame_path.split("\\")[-1])
                        except Exception:
                            st.warning("Frame not found.")
            else:
                st.info("No frames indexed yet. Start monitoring to save frames, then click Index frames.")

if __name__ == "__main__":
    main()