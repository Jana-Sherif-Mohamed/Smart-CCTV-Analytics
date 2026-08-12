import os, sys, tempfile
from datetime import datetime
import cv2
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.detection_model import DetectionModel, CLASS_NAMES
from core.tracker import Tracker
from core.analytics import AnalyticsEngine
from infrastructure.config_loader import load_config
from infrastructure.video_io import open_video, get_video_fps, create_video_writer


# Load YOLO model once
@st.cache_resource(show_spinner="Loading detection model...")
def get_detector(weights, confidence):
    cfg = load_config()
    cfg["model"]["weights"] = weights
    cfg["model"]["confidence"] = confidence
    model = DetectionModel(cfg)
    model.load()
    return model


# Page settings
st.set_page_config(
    page_title="Sentinel | Smart CCTV Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GUI styling
CUSTOM_CSS = """
<style>
.stApp {background:#FFF;}
#MainMenu,footer {visibility:hidden;}
.sentinel-header {display:flex;justify-content:space-between;align-items:center;padding:1rem 1.5rem;margin-bottom:1rem;border-radius:16px;background:#FFF;border:1px solid #EAECF0;}
.sentinel-title {font-size:1.5rem;font-weight:700;color:#1F2430;margin:0;}
.sentinel-subtitle {font-size:.85rem;color:#8A93A3;margin:2px 0 0;}
.status-pill {padding:6px 14px;border-radius:999px;font-size:.8rem;font-weight:600;}
.status-idle {background:#F1F3F6;color:#7C8494;border:1px solid #E2E5EA;}
.status-live {background:#E7F8EF;color:#17A566;border:1px solid #C7EFDA;}
.status-done {background:#EAF2FE;color:#3373E0;border:1px solid #D2E3FC;}
.section-label {font-size:.9rem;font-weight:700;color:#000!important;text-transform:uppercase;letter-spacing:.5px;margin-bottom:.4rem;border-left:3px solid #6FA8FF;padding-left:10px;}
div[data-testid="stMetric"] {background:#FFF;border:1px solid #EAECF0;padding:16px 18px;border-radius:14px;}
div[data-testid="stMetric"] * {color:#000!important;}
section[data-testid="stSidebar"] {background:#F5F7FA;border-right:1px solid #D9DDE5;}
section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] p {color:#000!important;}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {background:#FFF!important;color:#000!important;border-color:#D9DDE5!important;}
section[data-testid="stSidebar"] [data-baseweb="select"] span {color:#000!important;}
div[data-baseweb="popover"],div[data-baseweb="popover"] ul,div[data-baseweb="popover"] li {background:#FFF!important;}
div[data-baseweb="popover"] li {color:#000!important;}
div[data-baseweb="popover"] li:hover {background:#F1F3F6!important;}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {color:#000!important;}
.stButton > button {border-radius:10px;font-weight:600;}
.stButton > button[kind="primary"] {background:#6FA8FF;border:none;}
.stButton > button[kind="primary"]:hover {background:#5C97F5;}
hr {border-color:#EAECF0!important;}
</style>
"""


def render_header(status="idle"):
    """Display title and status."""
    status_map = {
        "idle": ("status-idle", "● IDLE"),
        "live": ("status-live", "● PROCESSING"),
        "done": ("status-done", "● COMPLETE")
    }
    css_class, label = status_map.get(status, status_map["idle"])
    st.markdown(f"""
    <div class="sentinel-header">
        <div>
            <p class="sentinel-title">🛡️ Sentinel — Smart CCTV Analytics</p>
            <p class="sentinel-subtitle">Real-time AI object detection, tracking & security analytics</p>
        </div>
        <span class="status-pill {css_class}">{label}</span>
    </div>
    """, unsafe_allow_html=True)


def resolve_video_source():
    """Get video from sample folder or upload."""
    st.sidebar.markdown('<p class="section-label">Video Source</p>', unsafe_allow_html=True)
    source = st.sidebar.radio("Input source", ["Select Sample Video", "Upload Video File"], label_visibility="collapsed")
    video_path, temp_path = None, None

    if source == "Select Sample Video":
        video_dir = "videos"
        files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4",".avi",".mov",".mkv"))] if os.path.exists(video_dir) else []
        if files:
            selected = st.sidebar.selectbox("Sample library", sorted(files))
            video_path = os.path.join(video_dir, selected)
    else:
        uploaded = st.sidebar.file_uploader("Upload footage", type=["mp4","avi","mov","mkv"], label_visibility="collapsed")
        if uploaded:
            suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp.write(uploaded.read())
            temp.close()
            temp_path = temp.name
            video_path = temp_path
            st.sidebar.success(f"Loaded: {uploaded.name}")

    return video_path, temp_path


def resolve_settings(config):
    """Get model and performance settings."""
    st.sidebar.markdown('<p class="section-label">Detection Settings</p>', unsafe_allow_html=True)
    models = ["yolo11n.pt","yolo11s.pt","yolo11m.pt"]
    current = config["model"].get("weights","yolo11s.pt")

    model = st.sidebar.selectbox("YOLO model", models, index=models.index(current) if current in models else 1)
    confidence = st.sidebar.slider("Confidence threshold", 0.10, 1.00, float(config["model"].get("confidence",0.40)), 0.05)
    detect_every = st.sidebar.slider("Run detection every N frames", 1, 15, 6)

    st.sidebar.markdown('<p class="section-label">Performance</p>', unsafe_allow_html=True)
    display_every = st.sidebar.slider("Refresh preview every N frames", 1, 10, 3)
    scale = st.sidebar.select_slider("Preview resolution", ["25%","50%","75%","100%"], value="50%")
    save_video = st.sidebar.checkbox("Save annotated output video", value=True)

    config["model"]["weights"] = model
    config["model"]["confidence"] = confidence

    return {
        "detect_every": detect_every,
        "display_every": display_every,
        "display_scale": int(scale[:-1])/100,
        "save_video": save_video
    }


def render_kpis(container):
    """Create KPI cards."""
    c1,c2,c3,c4 = container.columns(4)
    return {
        "people": c1.metric("Total People","0"),
        "vehicles": c2.metric("Total Vehicles","0"),
        "dwell": c3.metric("Avg Dwell Time","0s"),
        "alerts": c4.metric("Security Alerts","0")
    }


def update_kpis(kpis, stats):
    """Update KPI values."""
    kpis["people"].metric("Total People",stats["total_people"])
    kpis["vehicles"].metric("Total Vehicles",stats["total_vehicles"])
    kpis["dwell"].metric("Avg Dwell Time",f"{stats['average_dwell_time_sec']}s")
    kpis["alerts"].metric("Security Alerts",stats["total_alerts"])


def render_event_log(placeholder, events):
    """Display security events."""
    if events:
        placeholder.dataframe(pd.DataFrame(events[::-1]),height=380,use_container_width=True,hide_index=True)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "run_status" not in st.session_state:
        st.session_state.run_status = "idle"

    header = st.empty()
    with header.container():
        render_header(st.session_state.run_status)

    config = load_config()
    video_path, temp_path = resolve_video_source()
    options = resolve_settings(config)

    st.sidebar.markdown("---")
    start = st.sidebar.button("▶ Start Analytics",type="primary",use_container_width=True)
    st.sidebar.caption(f"Session started · {datetime.now().strftime('%H:%M:%S')}")

    # Overview
    st.markdown('<p class="section-label">Overview</p>',unsafe_allow_html=True)
    kpis = render_kpis(st.container())
    st.divider()

    # Main display
    video_col,log_col = st.columns([2,1])

    with video_col:
        st.markdown('<p class="section-label">Live Annotated Feed</p>',unsafe_allow_html=True)
        video_placeholder = st.empty()
        video_placeholder.info("Feed will appear here once analytics starts.")

    with log_col:
        st.markdown('<p class="section-label">Security Event Log</p>',unsafe_allow_html=True)
        log_placeholder = st.empty()
        log_placeholder.info("No security events detected yet.")

    if not start:
        return

    if not video_path or not os.path.exists(video_path):
        st.error("Please select or upload a valid video file.")
        return

    st.session_state.run_status = "live"
    with header.container():
        render_header("live")

    try:
        # Open video
        fps = get_video_fps(video_path)
        cap = open_video(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize AI
        detector = get_detector(config["model"]["weights"],config["model"]["confidence"])
        tracker = Tracker(CLASS_NAMES)
        analytics = AnalyticsEngine(config=config,fps=fps)

        detect_every = options["detect_every"]
        display_every = options["display_every"]
        display_scale = options["display_scale"]
        save_video = options["save_video"]

        # Create output video
        writer = None
        output_path = None

        if save_video:
            os.makedirs("outputs",exist_ok=True)
            output_path = os.path.join("outputs","streamlit_output.mp4")
            writer = create_video_writer(output_path,fps,width,height)

        progress = st.progress(0,text="Initializing...")
        frame_index = 0
        events_history = []
        last_annotated = None
        last_kpi_update = -1
        kpi_update_every = max(detect_every*3,1)

        # Process video
        while cap.isOpened():
            success,frame = cap.read()
            if not success:
                break

            is_detect_frame = frame_index % detect_every == 0

            if is_detect_frame:
                result = detector.predict(frame)
                tracked = tracker.update(result,frame_index)
                events = analytics.analyze(tracked,frame_index)

                for event in events:
                    events_history.append({
                        "Time (s)":f"{event.timestamp:.1f}s",
                        "Event Type":event.event_type,
                        "Track ID":event.track_id if event.track_id != -1 else "N/A",
                        "Message":event.message
                    })
                    render_event_log(log_placeholder,events_history)

                last_annotated = analytics.draw_zones(result.plot())

                if frame_index-last_kpi_update >= kpi_update_every:
                    update_kpis(kpis,analytics.summary())
                    last_kpi_update = frame_index

            # Save processed frame
            if writer is not None:
                writer.write(last_annotated if is_detect_frame else analytics.draw_zones(frame.copy()))

            # Update preview
            if last_annotated is not None and frame_index % display_every == 0:
                display = last_annotated
                if display_scale != 1.0:
                    display = cv2.resize(display,None,fx=display_scale,fy=display_scale,interpolation=cv2.INTER_AREA)
                display = cv2.cvtColor(display,cv2.COLOR_BGR2RGB)
                video_placeholder.image(display,channels="RGB",use_container_width=True)

            frame_index += 1

            if total_frames > 0 and frame_index % 5 == 0:
                progress.progress(min(frame_index/total_frames,1.0),text=f"Processing frame {frame_index}/{total_frames}")

        # Finish
        cap.release()
        if writer is not None:
            writer.release()

        progress.progress(1.0,text="Processing completed.")

        st.session_state.run_status = "done"
        with header.container():
            render_header("done")

        st.success("✅ Video processing completed successfully.")

        # Analytics summary
        st.markdown('<p class="section-label">Analytics Summary & Exports</p>',unsafe_allow_html=True)
        summary = analytics.summary()

        c1,c2,c3 = st.columns(3)
        c1.metric("Loitering Alerts",summary["loitering_events"])
        c2.metric("Restricted Zone Entries",summary["restricted_zone_events"])
        c3.metric("Crowding Events",summary["crowd_events"])

        col1,col2 = st.columns(2)

        with col1:
            with st.expander("Full summary (JSON)"):
                st.json({
                    "Total People":summary["total_people"],
                    "Total Vehicles":summary["total_vehicles"],
                    "Average Dwell Time (sec)":summary["average_dwell_time_sec"],
                    "Total Alerts":summary["total_alerts"],
                    "Loitering Alerts":summary["loitering_events"],
                    "Restricted Zone Entries":summary["restricted_zone_events"],
                    "Crowding Events":summary["crowd_events"]
                })

        with col2:
            if events_history:
                st.download_button(
                    "⬇ Download Security Events (CSV)",
                    data=pd.DataFrame(events_history).to_csv(index=False),
                    file_name="security_events_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            if output_path and os.path.exists(output_path):
                with open(output_path,"rb") as video:
                    st.download_button(
                        "⬇ Download Processed Video (MP4)",
                        data=video.read(),
                        file_name="annotated_cctv_output.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

    except Exception as error:
        # Handle errors
        st.session_state.run_status = "idle"
        st.error(f"An error occurred while processing the video: {error}")

    finally:
        # Delete temporary upload
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    main()