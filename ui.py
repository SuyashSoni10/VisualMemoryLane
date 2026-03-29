import streamlit as st
import cv2
import time
import os
import threading
from datetime import datetime
from PIL import Image
import numpy as np

from detector import Detector
from tracker import ObjectTracker
from context_engine import ContextEngine
from storage import init_db, search_objects, get_recent_logs, get_latest_llm, log_object

# Initialize DB on startup
init_db()

# Frame save interval (seconds)
FRAME_SAVE_INTERVAL = 300  # 5 minutes

def save_frame(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("frames", f"snapshot_{timestamp}.jpg")
    cv2.imwrite(path, frame)

def main():
    st.set_page_config(
        page_title="Visual Memory Lane",
        layout="wide"
    )

    st.title("Visual Memory Lane")
    st.caption("Context-aware visual AI assistant")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Live Feed", "Search", "Log"])

    # Shared state
    if "running" not in st.session_state:
        st.session_state.running = False
    if "suggestion" not in st.session_state:
        st.session_state.suggestion = "Waiting for first analysis..."
    if "suggestion_time" not in st.session_state:
        st.session_state.suggestion_time = None

    # --- TAB 1: Live Feed ---
    with tab1:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Camera Feed")
            frame_placeholder = st.empty()

        with col2:
            st.subheader("Scene State")
            scene_placeholder = st.empty()

        st.divider()
        st.subheader("AI Suggestion")
        suggestion_placeholder = st.empty()

        start_btn = st.button("Start", type="primary")
        stop_btn = st.button("Stop")

        if start_btn:
            st.session_state.running = True

        if stop_btn:
            st.session_state.running = False

        # Main detection loop
        if st.session_state.running:
            detector = Detector()
            tracker = ObjectTracker()
            engine = ContextEngine()
            detector.start()

            last_frame_save = time.time()

            while st.session_state.running:
                frame, detected_classes, detections = detector.get_frame()

                if frame is None:
                    st.error("Camera feed lost.")
                    break

                # Update tracker
                tracker.update(detected_classes)

                # Run context engine
                suggestion, suggestion_time = engine.run(tracker)

                # Save frame every 5 minutes
                if time.time() - last_frame_save >= FRAME_SAVE_INTERVAL:
                    save_frame(frame)
                    last_frame_save = time.time()

                # Display annotated frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, use_column_width=True)

                # Display scene state table
                scene_state = tracker.get_scene_state()
                if scene_state:
                    table_data = []
                    for obj, data in scene_state.items():
                        mins = data["duration_seconds"] // 60
                        table_data.append({
                            "Object": obj,
                            "Status": data["status"],
                            "Duration (mins)": mins,
                            "First Seen": data["first_seen"]
                        })
                    scene_placeholder.table(table_data)
                else:
                    scene_placeholder.info("No objects detected yet.")

                # Display LLM suggestion
                if suggestion:
                    suggestion_placeholder.info(
                        f"**{suggestion_time}** — {suggestion}"
                    )

                time.sleep(0.05)

            detector.stop()

    # --- TAB 2: Search ---
    with tab2:
        st.subheader("Search Object History")
        query = st.text_input("Search by object name (e.g. bottle, book, laptop)")

        if query:
            results = search_objects(query)
            if results:
                st.write(f"Found {len(results)} result(s):")
                for r in results:
                    st.markdown(f"""
                    - **{r[0]}** | First seen: {r[1]} | Last seen: {r[2]} | Duration: {r[3]//60} mins | Status: {r[4]}
                    """)
            else:
                st.warning("No results found.")

    # --- TAB 3: Log ---
    with tab3:
        st.subheader("Recent Action Log")

        if st.button("Refresh Log"):
            pass

        logs = get_recent_logs(20)
        if logs:
            for log in logs:
                st.markdown(f"- `{log[0]}` — **{log[1]}** — {log[2]}")
        else:
            st.info("No actions logged yet.")

        st.divider()
        st.subheader("Latest AI Suggestion")
        latest = get_latest_llm()
        if latest:
            st.success(f"**{latest[0]}** — {latest[1]}")
        else:
            st.info("No AI suggestions yet.")

if __name__ == "__main__":
    main()