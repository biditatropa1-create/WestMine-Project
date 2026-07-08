"""
WestMine Real-Time Worker Safety Monitoring System
ICT619 Assignment 2 - Streamlit Dashboard
Run with: streamlit run app.py

Authors: Bidita Tarafder, Tshering Wangmo, Cynthia Mosoba
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import gc
import platform
import tempfile
from datetime import datetime

# 5s timeout so a dead RTSP URL doesn't hang the dashboard.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000")

import config
from detector import SafetyDetector, FrameResult
from visualiser import draw_detections
from evaluator import SystemEvaluator

# ──────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=config.DASHBOARD_TITLE,
    page_icon=config.DASHBOARD_ICON,
    layout=config.PAGE_LAYOUT
)

# ──────────────────────────────────────────────
# CSS - works in both light and dark mode
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        transition: background 0.2s;
        color: #ccc !important;
        border-left: 3px solid transparent;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.12);
        border-left-color: rgba(231, 76, 60, 0.5);
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"] {
        background: rgba(231, 76, 60, 0.25);
        border-left: 3px solid #e74c3c;
        color: #fff !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #ccc !important;
    }
    section[data-testid="stSidebar"] .stMetric label {
        color: #aaa !important;
    }
    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: #fff !important;
    }

    /* ── Page Headers ── */
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e74c3c;
        margin-bottom: 1.2rem;
        color: #e74c3c !important;
    }
    .sub-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* ── Alert Cards ── */
    .alert-card {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border-left: 4px solid #e74c3c;
        margin-bottom: 0.5rem;
        color: #333;
    }
    .safe-card {
        background: linear-gradient(135deg, #f0fff0 0%, #e8ffe8 100%);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border-left: 4px solid #27ae60;
        color: #333;
    }
    .info-card {
        background: linear-gradient(135deg, #f0f8ff 0%, #e8f4fd 100%);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border-left: 4px solid #3498db;
        margin-bottom: 0.5rem;
        color: #333;
    }

    /* ── Status Badge ── */
    .status-safe {
        display: inline-block;
        background: #27ae60;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .status-danger {
        display: inline-block;
        background: #e74c3c;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* ── Team Cards ── */
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        color: white;
        transition: transform 0.3s, box-shadow 0.3s;
        cursor: default;
        min-height: 240px;
    }
    .team-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    }
    .team-avatar {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        background: rgba(255,255,255,0.2);
        margin: 0 auto 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 3px solid rgba(255,255,255,0.4);
        overflow: hidden;
    }
    .team-avatar img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 50%;
    }
    .team-email a {
        color: #ffffff !important;
        text-decoration: underline;
        font-size: 0.8rem;
        opacity: 0.9;
    }
    .team-name {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .team-role {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .team-id {
        font-size: 0.75rem;
        opacity: 0.65;
        margin-top: 0.3rem;
    }

    /* ── About Section Cards ── */
    .about-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        border: 1px solid #dee2e6;
        color: #333;
    }
    .about-card h4 {
        color: #e74c3c !important;
        margin-bottom: 0.5rem;
    }

    /* ── Chart width limit ── */
    .chart-container img {
        max-width: 700px !important;
    }

    /* ── Expander styling ── */
    .streamlit-expanderHeader {
        font-weight: 600;
    }

    /* ── Reset Session button - readable in light & dark mode ── */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        background: #8b1e1e !important;
        color: #ffffff !important;
        border: 1px solid #6a1414 !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background: #a82626 !important;
        color: #ffffff !important;
        border-color: #8b1e1e !important;
    }

    /* ── System status in sidebar ── */
    .sidebar-stat {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        text-align: center;
    }
    .sidebar-stat-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e74c3c !important;
    }
    .sidebar-stat-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #999 !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
# Anything that needs to persist across Streamlit reruns goes here.
if "detector" not in st.session_state:
    st.session_state.detector = None
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "danger_zone_points" not in st.session_state:
    st.session_state.danger_zone_points = config.DEFAULT_DANGER_ZONE
if "danger_zone_enabled" not in st.session_state:
    st.session_state.danger_zone_enabled = True
if "last_video_path" not in st.session_state:
    st.session_state.last_video_path = None
if "alert_frames_log" not in st.session_state:
    st.session_state.alert_frames_log = []
if "alert_frame_numbers" not in st.session_state:
    st.session_state.alert_frame_numbers = []

# Set True once the user edits the polygon so we don't reset to defaults.
if "zone_manually_edited" not in st.session_state:
    st.session_state.zone_manually_edited = False

# Stop flags for the live loops. The Stop buttons flip these and the
# loops check them each iteration so the user can stop mid-run.
if "processing_active" not in st.session_state:
    st.session_state.processing_active = False
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False
if "rtsp_active" not in st.session_state:
    st.session_state.rtsp_active = False

# Live settings written by sliders and read by the live loops.
if "live_confidence" not in st.session_state:
    st.session_state.live_confidence = config.CONFIDENCE_THRESHOLD
if "live_safe_distance" not in st.session_state:
    st.session_state.live_safe_distance = config.SAFE_DISTANCE_PX
if "live_cooldown" not in st.session_state:
    st.session_state.live_cooldown = config.ALERT_COOLDOWN_SECONDS
if "live_dist_multiplier" not in st.session_state:
    st.session_state.live_dist_multiplier = config.DISTANCE_MULTIPLIER
if "live_iou" not in st.session_state:
    st.session_state.live_iou = config.IOU_THRESHOLD
if "live_img_size" not in st.session_state:
    st.session_state.live_img_size = config.YOLO_IMG_SIZE
if "live_use_augment" not in st.session_state:
    st.session_state.live_use_augment = config.USE_AUGMENT
if "live_frame_skip" not in st.session_state:
    st.session_state.live_frame_skip = config.PROCESS_EVERY_N_FRAMES


def init_detector():
    """Load YOLOv8 once, reuse afterwards."""
    if st.session_state.detector is None:
        with st.spinner("Loading YOLOv8 model (first time may download weights)..."):
            try:
                st.session_state.detector = SafetyDetector()
            except Exception as e:
                st.error(
                    f"Failed to load YOLOv8 model: {e}\n\n"
                    f"Please check your internet connection and try again."
                )
                st.stop()
            if st.session_state.danger_zone_enabled:
                st.session_state.detector.set_danger_zone(
                    st.session_state.danger_zone_points
                )
    return st.session_state.detector


def apply_live_settings(detector):
    """Push current slider values into the detector before each run."""
    detector.confidence = st.session_state.live_confidence
    detector.iou_threshold = st.session_state.live_iou
    detector.safe_distance_px = st.session_state.live_safe_distance
    detector.distance_multiplier = st.session_state.live_dist_multiplier
    detector.alert_cooldown = st.session_state.live_cooldown
    detector.img_size = st.session_state.live_img_size
    detector.use_augment = st.session_state.live_use_augment


def cleanup_previous_temp_file():
    """Delete the previous uploaded video's temp file before making a new one."""
    old = st.session_state.get("last_video_path")
    if old and os.path.exists(old):
        try:
            gc.collect()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            os.unlink(old)
        except (PermissionError, OSError):
            pass
    st.session_state.last_video_path = None


def get_system_info():
    """Detect the current system hardware and software details."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "Unknown",
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import psutil
        info["ram"] = f"{round(psutil.virtual_memory().total / (1024**3), 1)} GB"
    except ImportError:
        info["ram"] = "Unknown"
    try:
        import torch
        info["pytorch"] = torch.__version__
        info["cuda"] = "Yes" if torch.cuda.is_available() else "No"
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        else:
            info["gpu"] = "N/A (CPU only)"
    except Exception:
        info["pytorch"] = "Unknown"
        info["cuda"] = "No"
        info["gpu"] = "N/A"
    return info


def extract_first_frame(video_path):
    """Pull the first frame from a video file for danger zone preview."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None


def draw_zone_preview(frame, points):
    """Draw the danger zone polygon on a frame for preview."""
    preview = frame.copy()
    if points and len(points) >= 3:
        pts = np.array(points, dtype=np.int32)
        overlay = preview.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 80))
        cv2.addWeighted(overlay, 0.3, preview, 0.7, 0, preview)
        cv2.polylines(preview, [pts], True, (0, 0, 200), 2)
        # Draw numbered corner dots so users can see which point is which
        for i, p in enumerate(points):
            cv2.circle(preview, p, 6, (0, 255, 0), -1)
            cv2.putText(preview, f"P{i+1}", (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def parse_zone_text(text):
    """Parse a 'x,y per line' string into a list of (x,y) tuples."""
    try:
        pts = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            x, y = line.split(",")
            pts.append((int(x.strip()), int(y.strip())))
        if len(pts) < 3:
            return None, "Need at least 3 points to make a polygon."
        return pts, None
    except Exception as e:
        return None, f"Invalid format: {e}"


def get_detector_counters():
    """Return (frames, alerts) from the detector if it's loaded, else 0,0."""
    det = st.session_state.detector
    if det is None:
        return 0, 0
    return det.total_frames, det.total_alerts


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    # Logo and title
    st.markdown("""
    <div style="text-align:center; padding: 0.8rem 0 0.5rem;">
        <div style="font-size: 2.5rem;">&#9935;</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #e74c3c !important;
                    letter-spacing: 1px;">WESTMINE</div>
        <div style="font-size: 0.7rem; color: #999; letter-spacing: 2px;
                    text-transform: uppercase;">Safety Monitor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar nav. Keep labels plain so the UI looks clean.
    nav_pages = [
        "Live Monitor",
        "Evaluation",
        "Settings",
        "Team",
        "About",
    ]
    page = st.radio(
        "Navigation",
        nav_pages,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # System status - cleaner display. Pulls directly from the detector
    # so the sidebar reflects the real model state (not a shadow counter).
    st.markdown("""
    <div style="text-align:center; margin-bottom: 4px;">
        <div class="sidebar-stat-label">System Status</div>
    </div>
    """, unsafe_allow_html=True)

    total_frames, total_alerts = get_detector_counters()
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-value">{total_frames}</div>
            <div class="sidebar-stat-label">Frames</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-value">{total_alerts}</div>
            <div class="sidebar-stat-label">Alerts</div>
        </div>
        """, unsafe_allow_html=True)

    # Reset button - clears alerts and detector counters between demo runs.
    if st.button("Reset Session", use_container_width=True):
        st.session_state.alert_log = []
        st.session_state.alert_frames_log = []
        st.session_state.alert_frame_numbers = []
        if st.session_state.detector is not None:
            st.session_state.detector.total_frames = 0
            st.session_state.detector.total_alerts = 0
            st.session_state.detector.reset_cooldowns()
        st.rerun()

    st.markdown("---")
    st.caption("ICT619 - Artificial Intelligence")
    st.caption("Murdoch University")
    st.caption(f"Model: {config.MODEL_NAME}")


# ══════════════════════════════════════════════
# PAGE: LIVE MONITOR
# ══════════════════════════════════════════════
if page == "Live Monitor":
    st.markdown('<div class="main-header">Real-Time Safety Monitor</div>',
                unsafe_allow_html=True)

    input_col1, input_col2 = st.columns([2, 1])
    with input_col1:
        input_type = st.selectbox(
            "Video Source",
            ["Upload Video File", "Webcam (Live)", "RTSP Stream"]
        )
    with input_col2:
        camera_id = st.text_input("Camera ID", value="CAM-01")

    # Detection settings - auto-collapsed once processing starts.
    settings_expanded = not st.session_state.processing_active
    with st.expander("Detection Settings", expanded=settings_expanded):
        st.markdown('<div class="info-card">Adjust these settings to fine-tune '
                    'how the system detects objects and measures distances. '
                    'Changes apply to the next run.</div>',
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.live_confidence = st.slider(
                "Confidence Threshold", 0.1, 1.0,
                st.session_state.live_confidence, 0.05,
                help="How sure the model must be before it counts a detection. "
                     "Lower = more detections but more false alarms. "
                     "Higher = fewer detections but more reliable. "
                     "0.5 (50%) matches our proposal."
            )
        with col2:
            st.session_state.live_safe_distance = st.slider(
                "Fallback Safe Distance (px)", 50, 500,
                st.session_state.live_safe_distance, 10,
                help="Minimum pixel distance between a person and vehicle "
                     "before an alert fires. This is only used as a fallback "
                     "when the dynamic distance calculation cannot determine "
                     "the vehicle size. Normally the system auto-calculates "
                     "safe distance based on how big the vehicle appears."
            )
        with col3:
            st.session_state.live_frame_skip = st.slider(
                "Process Every N Frames", 1, 10,
                st.session_state.live_frame_skip,
                help="Controls how many frames to skip between detections. "
                     "1 = process every frame (most accurate but slower). "
                     "3+ = skip frames (faster on older hardware but may "
                     "miss quick movements). Use higher values if your "
                     "computer is struggling to keep up."
            )

    # Danger zone toggle - hidden for webcam since the polygon doesn't make
    # sense on a free-roaming camera.
    if input_type != "Webcam (Live)":
        dz_col1, dz_col2 = st.columns([3, 1])
        with dz_col1:
            st.session_state.danger_zone_enabled = st.checkbox(
                "Enable Danger Zone",
                value=st.session_state.danger_zone_enabled,
                help="Turn the danger zone polygon on or off."
            )
    else:
        st.session_state.danger_zone_enabled = False

    # Video and metrics placeholders
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()

    # ── UPLOAD VIDEO ──
    if input_type == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload a video file (.mp4, .avi, .mov)",
            type=["mp4", "avi", "mov", "mkv"]
        )

        if uploaded_file is not None:
            # Clean up the previous upload's temp file before making a new one.
            # On Windows these don't get auto-deleted so they pile up.
            cleanup_previous_temp_file()

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            st.session_state.last_video_path = tfile.name

            first_frame = extract_first_frame(tfile.name)

            # Use the configured polygon directly. If the user hasn't edited
            # the points, reset to defaults from config.
            if not st.session_state.zone_manually_edited:
                st.session_state.danger_zone_points = list(config.DEFAULT_DANGER_ZONE)

            # Preview the danger zone on the first frame.
            if st.session_state.danger_zone_enabled:
                preview_expanded = not st.session_state.processing_active
                with st.expander("Danger Zone Preview - Adjust Zone on Your Video",
                                 expanded=preview_expanded):
                    st.markdown(
                        '<div class="info-card">Preview the danger zone on '
                        'your video. Edit the coordinates below (one '
                        'x,y per line) and click Update Zone to apply.</div>',
                        unsafe_allow_html=True
                    )

                    pv_col1, pv_col2 = st.columns([2, 1])
                    with pv_col2:
                        zone_edit = st.text_area(
                            "Polygon Points (x,y per line)",
                            value="\n".join(
                                f"{p[0]},{p[1]}"
                                for p in st.session_state.danger_zone_points
                            ),
                            height=180,
                            key="zone_preview_edit"
                        )
                        if st.button("Update Zone", key="preview_update"):
                            pts, err = parse_zone_text(zone_edit)
                            if err:
                                st.error(err)
                            else:
                                st.session_state.danger_zone_points = pts
                                st.session_state.zone_manually_edited = True
                                if st.session_state.detector:
                                    st.session_state.detector.set_danger_zone(pts)
                                st.success(f"Zone updated ({len(pts)} points)")
                                st.rerun()

                        if st.button("Reset to Default", key="preview_reset"):
                            st.session_state.danger_zone_points = list(
                                config.DEFAULT_DANGER_ZONE
                            )
                            st.session_state.zone_manually_edited = False
                            if st.session_state.detector:
                                st.session_state.detector.set_danger_zone(
                                    st.session_state.danger_zone_points
                                )
                            st.success("Reset to default.")
                            st.rerun()

                    with pv_col1:
                        if first_frame is not None:
                            preview = draw_zone_preview(
                                first_frame,
                                st.session_state.danger_zone_points
                            )
                            fh, fw = first_frame.shape[:2]
                            st.caption(f"Video resolution: {fw} x {fh} pixels")
                            st.image(preview, caption="First frame with danger zone",
                                     use_container_width=True)
                        else:
                            st.warning("Could not read the first frame of this video.")

            # Start/Stop buttons. Callbacks set state BEFORE script reruns
            # so the expanders above collapse immediately on first click.
            def _start_upload():
                st.session_state.processing_active = True

            def _stop_upload():
                st.session_state.processing_active = False

            start_col, stop_col = st.columns(2)
            start_clicked = start_col.button("Start Processing", type="primary",
                                             key="start_upload",
                                             on_click=_start_upload)
            stop_col.button("Stop Processing", key="stop_upload",
                            on_click=_stop_upload)

            if start_clicked:

                detector = init_detector()
                apply_live_settings(detector)
                detector.reset_cooldowns()

                # Apply danger zone setting
                if st.session_state.danger_zone_enabled:
                    detector.set_danger_zone(st.session_state.danger_zone_points)
                else:
                    detector.danger_zone = None
                    detector.danger_zone_points = []

                cap = None
                try:
                    cap = cv2.VideoCapture(tfile.name)
                    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_bar = st.progress(0)
                    frame_count = 0
                    # Target ~10 processed FPS by skipping based on video FPS.
                    # User's frame-skip slider still acts as a minimum.
                    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    target_fps = config.TARGET_PROCESS_FPS
                    auto_skip = max(1, int(round(video_fps / target_fps)))
                    frame_skip = max(auto_skip, st.session_state.live_frame_skip)

                    # Track alert frame numbers for the review slider
                    alert_frame_numbers = []

                    while cap.isOpened() and st.session_state.processing_active:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        if frame_count % frame_skip != 0:
                            continue

                        result = detector.detect(frame, camera_id=camera_id)

                        dz_pts = (detector.danger_zone_points
                                  if st.session_state.danger_zone_enabled else None)
                        annotated = draw_detections(frame, result, dz_pts)
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(annotated_rgb, channels="RGB",
                                                use_container_width=True)

                        with metrics_placeholder.container():
                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.metric("FPS", f"{result.fps:.1f}")
                            m2.metric("Persons", len(result.persons))
                            m3.metric("Vehicles", len(result.vehicles))
                            m4.metric("Frame", f"{frame_count}/{total_video_frames}")
                            if result.is_danger:
                                m5.markdown(
                                    '<div class="status-danger">DANGER</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                m5.markdown(
                                    '<div class="status-safe">SAFE</div>',
                                    unsafe_allow_html=True
                                )

                        if result.alerts:
                            for alert in result.alerts:
                                alert_frame_numbers.append(frame_count)
                                st.session_state.alert_log.append({
                                    "Time": alert.timestamp,
                                    "Camera": alert.camera_id,
                                    "Type": alert.alert_type.replace("_", " ").title(),
                                    "Distance (px)": f"{alert.distance_px:.0f}",
                                    "Confidence": f"{alert.confidence:.0%}",
                                    # Always str so column dtype is consistent.
                                    "Frame": str(frame_count),
                                })
                                # Save frame number and path for review
                                if alert.frame_path:
                                    st.session_state.alert_frames_log.append({
                                        "frame": frame_count,
                                        "time": alert.timestamp,
                                        "type": alert.alert_type.replace("_", " ").title(),
                                        "path": alert.frame_path,
                                    })

                        progress = frame_count / total_video_frames if total_video_frames > 0 else 0
                        progress_bar.progress(min(progress, 1.0))

                    # Save alert frame numbers for the review slider
                    st.session_state.alert_frame_numbers = alert_frame_numbers

                    if st.session_state.processing_active:
                        st.success(f"Done. {frame_count} frames processed, "
                                   f"{detector.total_alerts} total alerts so far.")
                    else:
                        st.warning(f"Stopped by user at frame {frame_count}.")

                except Exception as e:
                    st.error(f"Processing error: {e}")
                finally:
                    st.session_state.processing_active = False
                    if cap is not None:
                        cap.release()

        # ── VIDEO REVIEW SECTION ──
        # After processing, let the user jump to alert frames
        if st.session_state.alert_frames_log:
            st.markdown("---")
            st.markdown("### Review Alerts")
            st.markdown(
                '<div class="info-card">Select an alert below to see the '
                'saved frame. You can also click through each alert to '
                'verify if the detection was correct.</div>',
                unsafe_allow_html=True
            )

            # Build a list of alert options, filtering out any with
            # missing image files (happens if the output folder was cleaned)
            valid_alerts = [a for a in st.session_state.alert_frames_log
                            if os.path.exists(a["path"])]
            alert_options = [
                f"[Frame {a['frame']}] {a['time']} - {a['type']}"
                for a in valid_alerts
            ]

            if alert_options:
                selected_idx = st.selectbox(
                    "Jump to Alert",
                    range(len(alert_options)),
                    format_func=lambda i: alert_options[i],
                    key="alert_review_select"
                )

                if selected_idx is not None:
                    selected_alert = valid_alerts[selected_idx]
                    alert_img = cv2.imread(selected_alert["path"])
                    if alert_img is not None:
                        alert_img_rgb = cv2.cvtColor(alert_img, cv2.COLOR_BGR2RGB)
                        st.image(alert_img_rgb,
                                 caption=f"Alert at {selected_alert['time']} "
                                         f"- {selected_alert['type']}",
                                 use_container_width=True)
            else:
                st.info("Alert frames exist in the log but the saved images "
                        "could not be found on disk.")

    # ── WEBCAM ──
    elif input_type == "Webcam (Live)":
        st.info("This will use your computer's built-in camera for live monitoring.")

        start_col, stop_col = st.columns(2)
        start_click = start_col.button("Start Webcam", type="primary",
                                       key="start_webcam")
        stop_click = stop_col.button("Stop Webcam", key="stop_webcam")

        if stop_click:
            st.session_state.webcam_active = False

        if start_click:
            st.session_state.webcam_active = True

            detector = init_detector()
            apply_live_settings(detector)
            detector.reset_cooldowns()

            # Force 640 on webcam so it feels responsive on CPU.
            detector.img_size = 640

            # No danger zone for webcam.
            detector.danger_zone = None
            detector.danger_zone_points = []

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access the webcam. Make sure no other app "
                         "is using it, and try again.")
                st.session_state.webcam_active = False
            else:
                try:
                    while cap.isOpened() and st.session_state.webcam_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Lost connection to webcam.")
                            break

                        result = detector.detect(frame, camera_id=camera_id)
                        dz_pts = (detector.danger_zone_points
                                  if st.session_state.danger_zone_enabled else None)
                        annotated = draw_detections(frame, result, dz_pts)
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(annotated_rgb, channels="RGB",
                                               use_container_width=True)

                        with metrics_placeholder.container():
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("FPS", f"{result.fps:.1f}")
                            m2.metric("Persons", len(result.persons))
                            m3.metric("Vehicles", len(result.vehicles))
                            if result.is_danger:
                                m4.markdown('<div class="status-danger">DANGER</div>',
                                            unsafe_allow_html=True)
                            else:
                                m4.markdown('<div class="status-safe">SAFE</div>',
                                            unsafe_allow_html=True)

                        if result.alerts:
                            for alert in result.alerts:
                                st.session_state.alert_log.append({
                                    "Time": alert.timestamp,
                                    "Camera": alert.camera_id,
                                    "Type": alert.alert_type.replace("_", " ").title(),
                                    "Distance (px)": f"{alert.distance_px:.0f}",
                                    "Confidence": f"{alert.confidence:.0%}",
                                    "Frame": "LIVE",
                                })
                finally:
                    st.session_state.webcam_active = False
                    cap.release()
                    gc.collect()
                    try:
                        cv2.destroyAllWindows()
                    except Exception:
                        pass

    # ── RTSP STREAM ──
    elif input_type == "RTSP Stream":
        st.markdown(
            '<div class="info-card">Connect to a live CCTV camera using its '
            'RTSP URL. This is how real security cameras stream over a '
            'network.</div>',
            unsafe_allow_html=True
        )
        rtsp_url = st.text_input(
            "RTSP Stream URL",
            placeholder="rtsp://username:password@camera-ip:554/stream",
            help=(
                "Paste a real CCTV RTSP URL (most public test streams are "
                "unreliable). Format: rtsp://user:pass@host:port/path. "
                "OpenCV has a 5s timeout so dead URLs don't freeze the app."
            ),
        )

        start_col, stop_col = st.columns(2)
        start_rtsp = start_col.button("Connect Stream", type="primary",
                                      key="start_rtsp")
        stop_rtsp = stop_col.button("Stop Stream", key="stop_rtsp")

        if stop_rtsp:
            st.session_state.rtsp_active = False

        if rtsp_url and start_rtsp:
            st.session_state.rtsp_active = True

            detector = init_detector()
            apply_live_settings(detector)
            detector.reset_cooldowns()

            if st.session_state.danger_zone_enabled:
                detector.set_danger_zone(st.session_state.danger_zone_points)
            else:
                detector.danger_zone = None
                detector.danger_zone_points = []

            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                st.error(f"Cannot connect to stream: {rtsp_url}\n"
                         "Check the URL, network connection, and camera status.")
                st.session_state.rtsp_active = False
            else:
                st.success("Connected to RTSP stream.")
                try:
                    while cap.isOpened() and st.session_state.rtsp_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Stream ended or connection lost.")
                            break

                        result = detector.detect(frame, camera_id=camera_id)
                        dz_pts = (detector.danger_zone_points
                                  if st.session_state.danger_zone_enabled else None)
                        annotated = draw_detections(frame, result, dz_pts)
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(annotated_rgb, channels="RGB",
                                               use_container_width=True)

                        with metrics_placeholder.container():
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("FPS", f"{result.fps:.1f}")
                            m2.metric("Persons", len(result.persons))
                            m3.metric("Vehicles", len(result.vehicles))
                            if result.is_danger:
                                m4.markdown('<div class="status-danger">DANGER</div>',
                                            unsafe_allow_html=True)
                            else:
                                m4.markdown('<div class="status-safe">SAFE</div>',
                                            unsafe_allow_html=True)

                        if result.alerts:
                            for alert in result.alerts:
                                st.session_state.alert_log.append({
                                    "Time": alert.timestamp,
                                    "Camera": alert.camera_id,
                                    "Type": alert.alert_type.replace("_", " ").title(),
                                    "Distance (px)": f"{alert.distance_px:.0f}",
                                    "Confidence": f"{alert.confidence:.0%}",
                                    "Frame": "LIVE",
                                })
                finally:
                    st.session_state.rtsp_active = False
                    cap.release()
                    gc.collect()

    # ── ALERT LOG TABLE ──
    st.markdown("---")
    st.markdown("### Alert Event Log")
    st.markdown(
        '<div class="info-card">Every alert is logged with a timestamp and '
        'details. This creates an auditable safety record as required by '
        'WHS Regulation 621.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.alert_log:
        df = pd.DataFrame(st.session_state.alert_log)

        # Make sure every column exists and Frame is a string.
        for col in ["Time", "Camera", "Type", "Distance (px)",
                    "Confidence", "Frame"]:
            if col not in df.columns:
                df[col] = "N/A"
        df["Frame"] = df["Frame"].astype(str)

        # Filter by alert type.
        alert_types = ["All"] + sorted(
            [t for t in df["Type"].dropna().unique().tolist()]
        )
        type_filter = st.selectbox("Filter by type", alert_types, index=0,
                                    key="alert_type_filter")
        if type_filter != "All":
            df_display = df[df["Type"] == type_filter]
        else:
            df_display = df

        st.dataframe(df_display, use_container_width=True)

        csv = df_display.to_csv(index=False)
        st.download_button(
            label="Download Alert Log (CSV)",
            data=csv,
            file_name=f"westmine_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.markdown(
            '<div class="safe-card">No alerts recorded yet. '
            'Upload a video or start a camera to begin monitoring.</div>',
            unsafe_allow_html=True
        )

    # Heatmap of alert positions for the current session.
    st.markdown("---")
    st.markdown("### Alert Hotspot Heatmap")
    st.markdown(
        '<div class="info-card">A heatmap of every alert position from the '
        'current session. Red areas are where danger has occurred most '
        'often. Clear the data with the Reset Session button in the sidebar.'
        '</div>',
        unsafe_allow_html=True
    )
    det_for_heatmap = st.session_state.detector
    if det_for_heatmap is not None and det_for_heatmap.alert_positions:
        try:
            import matplotlib.pyplot as plt
            positions = det_for_heatmap.alert_positions
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist2d(xs, ys, bins=20, cmap='hot')
            ax.set_xlabel("Frame X (pixels)")
            ax.set_ylabel("Frame Y (pixels)")
            ax.set_title(f"Alert positions ({len(positions)} alerts)")
            ax.invert_yaxis()   # image coords have y=0 at top
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
        except Exception as heatmap_err:
            st.warning(f"Could not draw heatmap: {heatmap_err}")
    else:
        st.info("No alerts captured yet. Run detection on a video with "
                "workers near vehicles to populate the heatmap.")


# ══════════════════════════════════════════════
# PAGE: EVALUATION
# ══════════════════════════════════════════════
elif page == "Evaluation":
    st.markdown('<div class="main-header">System Evaluation</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="info-card">This page shows how well the detection model '
        'performs. All metrics come from actual model inference on test data, '
        'not simulated values.</div>',
        unsafe_allow_html=True
    )

    # COCO benchmarks with info tooltip
    with st.expander("YOLOv8s Published COCO Benchmarks (Reference)", expanded=True):
        st.markdown(
            "These numbers are from the official Ultralytics benchmarks on the "
            "COCO val2017 dataset. They are reference values only - our actual "
            "performance on mining footage may differ due to different camera "
            "angles, lighting, object sizes, and the fact that mining vehicles "
            "are visually different from typical cars in the COCO dataset."
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("mAP@0.5", "52.6%",
                     help="Mean Average Precision at IoU 0.5. Higher is "
                          "better. 52.6% is the published value for YOLOv8s.")
        col2.metric("mAP@0.5:0.95", "44.9%",
                     help="Stricter mAP averaged across IoU 0.5 to 0.95.")
        col3.metric("Parameters", "11.2M",
                     help="Trainable parameters in YOLOv8s.")
        col4.metric("FLOPs", "28.6B",
                     help="Floating Point Operations per inference.")

    st.markdown("---")

    # Run evaluation
    st.markdown("### Run Evaluation on Your Video")
    st.markdown(
        "Upload a test video and the system will run YOLOv8 on it, "
        "measuring real inference speed, confidence scores, and "
        "detection counts."
    )

    eval_video = st.file_uploader("Upload test video", type=["mp4", "avi", "mov"],
                                  key="eval_video")

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        eval_frames = st.slider(
            "Max frames to test", 50, 500, 200, 50,
            help="How many frames to sample from the video. More frames "
                 "gives more reliable averages but takes longer."
        )
    with eval_col2:
        eval_sample = st.slider(
            "Sample every N frames", 1, 20, 5,
            help="Process every Nth frame. Higher values skip more frames "
                 "and cover more of the video timeline but test fewer total."
        )

    if eval_video and st.button("Run Evaluation", type="primary"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(eval_video.read())
        tfile.close()

        try:
            with st.spinner("Running evaluation (this may take a few minutes)..."):
                evaluator = SystemEvaluator()
                metrics = evaluator.evaluate_on_video(
                    tfile.name,
                    sample_rate=eval_sample,
                    max_frames=eval_frames
                )

                # Run ground-truth eval first so precision/recall get
                # saved to the metrics JSON from generate_report()
                if os.path.exists(config.GROUND_TRUTH_CSV):
                    gt_results = evaluator.evaluate_against_ground_truth(
                        config.GROUND_TRUTH_CSV, tfile.name
                    )
                else:
                    gt_results = None

                evaluator.generate_report()

            st.success("Evaluation complete! All metrics are from real model inference.")

            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Frames Tested", metrics.total_frames,
                        help="Total number of frames the model processed.")
            col2.metric("Avg FPS", f"{metrics.avg_fps:.1f}",
                        help="Average frames per second. This tells you how "
                             "fast the model runs on your hardware.")
            col3.metric("Avg Inference", f"{metrics.avg_inference_ms:.0f}ms",
                        help="Average time to process one frame in "
                             "milliseconds. Lower is faster.")
            col4.metric("Avg Confidence", f"{metrics.avg_confidence:.2%}",
                        help="Average confidence score across all detections. "
                             "Higher means the model is more certain about "
                             "what it found.")

            # Additional stats
            st.markdown("---")
            st.markdown("### Detection Breakdown")
            det_col1, det_col2, det_col3, det_col4 = st.columns(4)
            det_col1.metric("Total Persons", evaluator.detection_counts.get("Person", 0))
            det_col2.metric("Total Cars", evaluator.detection_counts.get("Car", 0))
            det_col3.metric("Total Trucks", evaluator.detection_counts.get("Truck", 0))
            total_dets = sum(evaluator.detection_counts.values())
            det_col4.metric("Total Detections", total_dets)

            # Environment info
            st.markdown("---")
            st.markdown("### Environment Setup")
            st.markdown(
                '<div class="info-card">Hardware and software detected on '
                'the machine running this evaluation.</div>',
                unsafe_allow_html=True
            )
            env_info = evaluator.get_environment_info()
            env_col1, env_col2 = st.columns(2)
            with env_col1:
                st.write(f"**OS:** {env_info.get('os', 'Unknown')}")
                st.write(f"**CPU:** {env_info.get('cpu', 'Unknown')}")
                st.write(f"**RAM:** {env_info.get('ram_gb', 'Unknown')} GB")
                st.write(f"**GPU:** {env_info.get('gpu_name', 'N/A')}")
            with env_col2:
                st.write(f"**Python:** {env_info.get('python_version', 'Unknown')}")
                st.write(f"**PyTorch:** {env_info.get('pytorch_version', 'Unknown')}")
                st.write(f"**CUDA Available:** {env_info.get('cuda_available', False)}")
                st.write(f"**Ultralytics:** {env_info.get('ultralytics_version', 'Unknown')}")

            # Ground truth comparison
            if gt_results is not None:
                st.markdown("---")
                st.markdown("### Ground Truth Comparison")
                gcol1, gcol2, gcol3, gcol4 = st.columns(4)
                gcol1.metric("Precision", f"{gt_results['precision']:.2%}",
                             help="Of all the times the model said 'danger', "
                                  "how often was it actually correct?")
                gcol2.metric("Recall", f"{gt_results['recall']:.2%}",
                             help="Of all the real danger situations, how "
                                  "many did the model catch?")
                gcol3.metric("F1 Score", f"{gt_results['f1_score']:.2%}",
                             help="Balanced score combining precision and "
                                  "recall. Higher is better.")
                target_met = "Yes" if gt_results['meets_target_recall'] else "No"
                gcol4.metric("Recall >= 90%?", target_met,
                             help="Our target is 90% recall because missing "
                                  "a real danger is worse than a false alarm.")

            # Let users download the metrics JSON directly from the dashboard
            metrics_json_path = os.path.join(config.EVAL_DIR,
                                              "evaluation_metrics.json")
            if os.path.exists(metrics_json_path):
                with open(metrics_json_path, "rb") as f:
                    st.download_button(
                        "Download Metrics JSON",
                        data=f.read(),
                        file_name="evaluation_metrics.json",
                        mime="application/json"
                    )

        except Exception as e:
            st.error(f"Evaluation error: {e}")
        finally:
            try:
                if os.path.exists(tfile.name):
                    os.unlink(tfile.name)
            except (PermissionError, OSError):
                pass

    # Charts section with width limit
    st.markdown("---")
    st.markdown("### Evaluation Charts")
    st.markdown(
        '<div class="info-card">These charts are generated from real model '
        'data collected during evaluation runs.</div>',
        unsafe_allow_html=True
    )

    chart_files = [
        ("confidence_distribution.png",
         "Detection Confidence Distribution",
         "Shows how confident the model was for each detection. "
         "Most detections should cluster above the 0.5 threshold line. "
         "A spread towards 1.0 means the model is very sure."),
        ("detection_classes.png",
         "Detections by Class",
         "Shows how many of each object type the model found. "
         "This helps verify the model is detecting the right things "
         "in your specific video footage."),
        ("fps_over_time.png",
         "Inference Speed Over Time",
         "Shows how fast the model processed each frame. Dips in "
         "speed usually happen on frames with many objects. The red "
         "line shows the average FPS across the whole run."),
    ]

    for filename, title, explanation in chart_files:
        path = os.path.join(config.EVAL_DIR, filename)
        if os.path.exists(path):
            with st.expander(title, expanded=True):
                st.markdown(f"*{explanation}*")
                # Use columns to limit chart width
                ch_col1, ch_col2 = st.columns([3, 1])
                with ch_col1:
                    st.image(path, use_container_width=True)

    if not any(os.path.exists(os.path.join(config.EVAL_DIR, f[0]))
               for f in chart_files):
        st.info("Upload a video above and click 'Run Evaluation' to generate "
                "charts from real detection data.")


# ══════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════
elif page == "Settings":
    st.markdown('<div class="main-header">System Settings</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="info-card">Settings here are applied to the live '
        'detector the next time you run detection. Click '
        '<strong>Apply Settings</strong> at the bottom to push the values '
        'into the running detector immediately.</div>',
        unsafe_allow_html=True
    )

    # Model configuration
    st.markdown("### Model Configuration")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Model", config.MODEL_NAME, disabled=True,
                       help="YOLOv8s (small variant). We chose it over the "
                            "nano model for better accuracy on small/distant "
                            "objects, while still running on CPU.")
        st.session_state.live_confidence = st.slider(
            "Confidence Threshold", 0.1, 1.0,
            st.session_state.live_confidence, 0.05, key="s_conf",
            help="Minimum confidence for a detection to count. "
                 "Sliding LEFT (lower) = more detections but more false alarms. "
                 "Sliding RIGHT (higher) = fewer detections but more reliable."
        )
    with col2:
        st.text_input("Target Classes", "Person (0), Car (2), Truck (7)",
                      disabled=True,
                      help="COCO class IDs that the system looks for. "
                           "0 = person, 2 = car, 7 = truck. These are "
                           "built into the COCO dataset YOLOv8 was trained on.")
        st.session_state.live_iou = st.slider(
            "NMS IoU Threshold", 0.1, 1.0,
            st.session_state.live_iou, 0.05, key="s_iou",
            help="Non-Maximum Suppression threshold. Controls how much "
                 "overlap is allowed between detection boxes. "
                 "Sliding LEFT = stricter, removes more overlapping boxes. "
                 "Sliding RIGHT = more permissive, keeps overlapping boxes."
        )

    # Advanced detection settings (these address the "missed detections" issue)
    st.markdown("---")
    st.markdown("### Detection Quality")
    st.markdown(
        '<div class="info-card">These settings trade speed for accuracy. '
        'If the system is missing small/far-away workers or trucks, '
        'try raising the image size or enabling augmentation.</div>',
        unsafe_allow_html=True
    )
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        st.session_state.live_img_size = st.select_slider(
            "YOLO Image Size",
            options=[320, 480, 640, 800, 960, 1280, 1600],
            value=st.session_state.live_img_size,
            help="Input resolution for the model. The default YOLO value is "
                 "640 but we use 1280 so tiny workers in the distance are "
                 "still detected. Bigger = more accurate but slower."
        )
    with adv_col2:
        st.session_state.live_use_augment = st.checkbox(
            "Test-Time Augmentation (slower, better recall)",
            value=st.session_state.live_use_augment,
            help="Run YOLO on a few flipped/scaled versions of each frame "
                 "and merge the results. Usually catches a few extra "
                 "detections but roughly halves FPS."
        )

    # Alert settings
    st.markdown("---")
    st.markdown("### Alert Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.live_cooldown = st.slider(
            "Alert Cooldown (seconds)", 1, 30,
            st.session_state.live_cooldown, 1, key="s_cooldown",
            help="Minimum time between alerts for the same person/vehicle "
                 "pair. Different people in the same scene still get their "
                 "own alerts without waiting for the cooldown."
        )
        st.session_state.live_dist_multiplier = st.slider(
            "Distance Multiplier", 1.0, 3.0,
            st.session_state.live_dist_multiplier, 0.1, key="s_dist_mult",
            help="How many times the vehicle width to use as safe distance. "
                 "1.5 means the safe zone around a vehicle extends to 1.5x "
                 "its visible width. Higher = larger safe zone = more alerts."
        )
    with col2:
        st.checkbox(
            "Save Alert Frames", value=config.SAVE_ALERT_FRAMES,
            key="s_save_frames", disabled=True,
            help="Save the actual video frame as a JPEG image when an alert "
                 "triggers. Controlled by SAVE_ALERT_FRAMES in config.py."
        )
        st.checkbox(
            "Simulate SMS Alerts", value=config.SIMULATE_SMS,
            key="s_sms", disabled=True,
            help="Print a simulated SMS message to the console when an alert "
                 "fires. Controlled by SIMULATE_SMS in config.py."
        )

    # Video processing settings
    st.markdown("---")
    st.markdown("### Video Processing")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.live_frame_skip = st.slider(
            "Frame Skip (Process Every N)", 1, 10,
            st.session_state.live_frame_skip, 1, key="s_frameskip",
            help="Skip frames to speed up processing on slower hardware. "
                 "1 = every frame (best accuracy). "
                 "3 = every 3rd frame (3x faster but may miss things)."
        )
    with col2:
        st.number_input(
            "Max Display FPS", 1, 60,
            config.MAX_FPS_DISPLAY, key="s_maxfps", disabled=True,
            help="Display-only refresh cap. The detector runs as fast as it "
                 "can; this only affects how often Streamlit repaints."
        )

    # Apply button - pushes the current slider values into the running
    # detector so changes take effect without restarting the app
    st.markdown("---")
    if st.button("Apply Settings to Detector", type="primary"):
        if st.session_state.detector is not None:
            apply_live_settings(st.session_state.detector)
            st.success("Settings applied to the running detector.")
        else:
            st.info("Settings saved. They'll be applied when you start "
                    "detection on the Live Monitor page.")

    st.markdown("---")

    # Danger zone editor
    st.markdown("### Danger Zone Configuration")
    st.markdown(
        '<div class="info-card">Edit the polygon coordinates below to change '
        'the danger zone shape. Each line is one corner of the polygon in '
        'the format x,y (pixels). These would be recalibrated for each camera '
        'at a real site. Use the Live Monitor page to preview the zone on '
        'your actual video.</div>',
        unsafe_allow_html=True
    )

    dz_enabled = st.checkbox(
        "Danger Zone Enabled",
        value=st.session_state.danger_zone_enabled,
        key="settings_dz_enabled",
        help="Toggle the danger zone on or off. Disable for moving/PTZ cameras."
    )
    st.session_state.danger_zone_enabled = dz_enabled

    if dz_enabled:
        # Explicit key so this widget doesn't share state with the Live
        # Monitor's zone editor.
        zone_text = st.text_area(
            "Polygon Points (one per line: x,y)",
            value="\n".join(
                f"{p[0]},{p[1]}"
                for p in st.session_state.danger_zone_points
            ),
            height=140,
            key="settings_zone_editor"
        )

        if st.button("Update Danger Zone", key="settings_update_zone"):
            pts, err = parse_zone_text(zone_text)
            if err:
                st.error(err)
            else:
                st.session_state.danger_zone_points = pts
                st.session_state.zone_manually_edited = True
                if st.session_state.detector:
                    st.session_state.detector.set_danger_zone(pts)
                st.success(f"Danger zone updated ({len(pts)} points).")
                # Rerun so the text area refreshes on the first click.
                st.rerun()

        if st.button("Reset to Default Polygon", key="settings_reset_zone"):
            st.session_state.danger_zone_points = list(config.DEFAULT_DANGER_ZONE)
            st.session_state.zone_manually_edited = False
            if st.session_state.detector:
                st.session_state.detector.set_danger_zone(
                    st.session_state.danger_zone_points
                )
            st.success("Danger zone reset to default.")
            st.rerun()

    st.markdown("---")

    # Live system hardware info (replaces static profiles)
    st.markdown("### Current System Hardware")
    st.markdown(
        '<div class="info-card">Automatically detected hardware and software '
        'on this computer. This information helps you understand what '
        'performance to expect.</div>',
        unsafe_allow_html=True
    )

    sys_info = get_system_info()
    hw_col1, hw_col2 = st.columns(2)
    with hw_col1:
        st.write(f"**Operating System:** {sys_info['os']}")
        st.write(f"**CPU:** {sys_info['cpu']}")
        st.write(f"**Architecture:** {sys_info['machine']}")
        st.write(f"**RAM:** {sys_info['ram']}")
    with hw_col2:
        st.write(f"**Python:** {sys_info['python']}")
        st.write(f"**PyTorch:** {sys_info['pytorch']}")
        st.write(f"**CUDA (GPU):** {sys_info['cuda']}")
        st.write(f"**GPU:** {sys_info['gpu']}")

    # Performance recommendations based on detected hardware
    if sys_info['cuda'] == "Yes":
        st.markdown(
            '<div class="safe-card">GPU detected! You should get good '
            'performance (15+ FPS). Set frame skip to 1 for best accuracy.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="alert-card">No GPU detected. Running on CPU only. '
            'Expect 2-5 FPS with YOLOv8s. If performance is slow, try '
            'increasing the frame skip to 3 or higher in Detection Settings.'
            '</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════
# PAGE: TEAM
# ══════════════════════════════════════════════
elif page == "Team":
    st.markdown('<div class="main-header">Team Members</div>',
                unsafe_allow_html=True)

    st.markdown(
        "ICT619 - Artificial Intelligence | Assignment 2 | Murdoch University"
    )
    st.markdown("---")

    # Group members - flowers as avatars (Unsplash direct image URLs).
    col1, col2, col3 = st.columns(3)

    members = [
        {
            "name": "Bidita Tarafder",
            "id": "35315146",
            "email": "35315146@student.murdoch.edu.au",
            "image": "https://images.unsplash.com/photo-1518895949257-7621c3c786d7?w=300&h=300&fit=crop",
        },
        {
            "name": "Tshering Wangmo",
            "id": "35410453",
            "email": "35410453@student.murdoch.edu.au",
            "image": "https://images.unsplash.com/photo-1470509037663-253afd7f0f51?w=300&h=300&fit=crop",
        },
        {
            "name": "Cynthia Mosoba",
            "id": "35434937",
            "email": "35434937@student.murdoch.edu.au",
            "image": "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=300&h=300&fit=crop",
        },
    ]

    for col, member in zip([col1, col2, col3], members):
        with col:
            st.markdown(f"""
            <div class="team-card">
                <div class="team-avatar"><img src="{member['image']}" alt="{member['name']}"/></div>
                <div class="team-name">{member['name']}</div>
                <div class="team-id">Student ID: {member['id']}</div>
                <div class="team-email"><a href="mailto:{member['email']}">{member['email']}</a></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Contributions")
    st.markdown(
        "All three members were equal partners on this project. The workload "
        "was shared evenly across every task — research, project planning, "
        "writing the proposal, coding the system, writing the final report, "
        "and preparing the PowerPoint presentation."
    )


# ══════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════
elif page == "About":
    st.markdown('<div class="main-header">About This System</div>',
                unsafe_allow_html=True)

    # Hero section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: white; border-radius: 12px; padding: 2rem; margin-bottom: 1.5rem;">
        <h2 style="color: #e74c3c; margin-top: 0;">WestMine Real-Time Worker
        Safety Monitoring System</h2>
        <p style="color: #ccc; margin-bottom: 0.5rem;">
            <strong>Course:</strong> ICT619 - Artificial Intelligence &nbsp;|&nbsp;
            <strong>Institution:</strong> Murdoch University &nbsp;|&nbsp;
            <strong>Assignment:</strong> 2 - AI Solution Implementation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # What it does
    st.markdown("""
    <div class="about-card">
        <h4>What This System Does</h4>
        <p>This system uses YOLOv8 computer vision to analyse CCTV footage and detect
        when a worker on foot enters a restricted zone or gets dangerously close
        to heavy vehicles. When danger is detected, it alerts the supervisor
        through the dashboard and logs the event with a timestamp and saved frame.</p>
    </div>
    """, unsafe_allow_html=True)

    # Key features in two columns
    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.markdown("""
        <div class="about-card">
            <h4>Detection Features</h4>
            <p>Real-time object detection using YOLOv8s (small variant, runs on CPU).
            Danger zone monitoring with customisable polygon boundaries.
            Dynamic proximity thresholding that adapts to camera perspective.
            Feet-based zone intrusion checking (checks where person is standing).
            Per-person/per-vehicle cooldown so multiple workers in the same scene
            all trigger their own alerts.</p>
        </div>
        """, unsafe_allow_html=True)
    with feat_col2:
        st.markdown("""
        <div class="about-card">
            <h4>Safety and Compliance</h4>
            <p>Simulated SMS alerts and timestamped event logging.
            CSV export of alert log for compliance auditing.
            Saved alert frame images for visual evidence.
            Real evaluation metrics from actual model inference.
            Alert hotspot heatmap to identify recurring danger areas.
            Model warmup on startup so the first processed frame does not
            stall.</p>
        </div>
        """, unsafe_allow_html=True)

    # Technical stack
    st.markdown("""
    <div class="about-card">
        <h4>Technical Stack</h4>
    </div>
    """, unsafe_allow_html=True)

    tech_data = {
        "Component": ["Object Detection", "Pre-trained Data", "Video Processing",
                      "Geometry Analysis", "Dashboard", "Evaluation"],
        "Technology": ["YOLOv8s by Ultralytics", "COCO dataset (330K images, 80 classes)",
                      "OpenCV (cv2)", "Shapely (polygon and point checks)",
                      "Streamlit", "matplotlib, pandas"],
    }
    st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

    # Changes from proposal
    with st.expander("Changes from Assignment 1 Proposal", expanded=False):
        st.markdown("""
        During implementation, we made some adjustments to the methodology
        proposed in Assignment 1. The core problem (worker-vehicle proximity
        detection) remains the same, but a few details were refined:

        **1. Feet-Based Zone Check (Improved from Proposal)**

        The proposal described using Intersection over Union (IoU) between
        bounding boxes and the danger zone polygon. During testing, we found
        that a person standing just outside the zone could have their head
        or shoulders overlapping the zone in the 2D camera view due to
        perspective. So instead, we check whether the person's FEET
        (bottom-centre of their bounding box) are inside the polygon.

        **2. Dynamic Distance Thresholding (Improved from Proposal)**

        The proposal mentioned a fixed pixel threshold calibrated to a
        real-world safe distance. We improved this by calculating the safe
        distance dynamically based on the vehicle's apparent size in the
        frame. A vehicle far from the camera appears smaller, so the safe
        distance threshold automatically scales down with it.

        **3. YOLOv8s and Larger Image Size (Updated)**

        The proposal mentioned YOLOv8n. After testing we switched to YOLOv8s
        (small variant) for better accuracy on small/distant objects, and
        increased the input size from the default 640 to 1280 to catch more
        far-away workers and trucks. Both still run on CPU.

        **4. Per-Person Alert Cooldown (Fix)**

        Our first version had a single global cooldown which meant that
        if two workers were in danger simultaneously, only one alert fired.
        That was bad for safety, so we changed it to a per-(person, vehicle)
        cooldown so every distinct danger situation gets its own alert.

        **5. Grad-CAM Explainability (Removed)**

        In Assignment 1 we proposed using Grad-CAM heatmaps to explain why the
        model flagged a frame. During implementation, we found that YOLOv8's
        anchor-free architecture does not work well with standard Grad-CAM
        libraries without unstable workarounds. We removed this feature to keep
        the system stable and reliable.
        """)

    # Known limitations
    with st.expander("Known Limitations", expanded=False):
        st.markdown("""
        **1. 2D Pixel Distance (Not True 3D)** - Our proximity check uses
        Euclidean distance in 2D pixels. Even with dynamic thresholding,
        this is not a true 3D measurement. A full production system would
        use a perspective transform (homography matrix).

        **2. CPU Performance** - YOLOv8s runs at 2-5 FPS on CPU hardware.
        This is sufficient for a proof-of-concept but a real deployment would
        need GPU hardware for smooth real-time processing at 25+ FPS.

        **3. Ground Truth Evaluation** - Full precision/recall evaluation
        requires manually annotated test frames. A ground_truth_TEMPLATE.csv
        is included in output/evaluation/ as a starting point for annotation.

        **4. Lighting and Weather Conditions** - Detection accuracy drops
        in heavy dust, rain, low lighting, and partial occlusion.

        **5. Low Resolution Video** - Videos below 480p may not have enough
        detail for the model to detect people and vehicles reliably,
        especially when objects are far from the camera.
        """)

    # Regulatory alignment
    st.markdown("""
    <div class="about-card">
        <h4>Regulatory Alignment</h4>
        <p>WHS (Mines) Regulations 2022 (WA) - Regulations 617 and 621.
        Australia's AI Ethics Principles (DISR, 2019) - Principles 6 and 7.
        Voluntary AI Safety Standard (DISR, 2024) - Guardrail 5.</p>
    </div>
    """, unsafe_allow_html=True)

