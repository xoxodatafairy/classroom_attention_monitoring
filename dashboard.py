"""
dashboard.py

AI Classroom Attention Monitoring System

Tabs
----
Dashboard   - alert banner, KPI cards, performance gauge, behaviour
              distribution, rPPG pulse waveform
Monitoring  - input sources (video upload, image, camera snapshot, webcam,
              CCTV), attention charts, frame viewer, heat map
Students    - registered students
Teacher     - teacher profile and classroom insights
Reports     - summary tables and CSV downloads
Register    - register a student

Run from the project root:
    streamlit run dashboard.py
"""

import importlib
import inspect
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ==================================================
# PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "Backend"

MODEL_PATH = BACKEND / "models" / "best.pt"
DEFAULT_VIDEO = BACKEND / "videos" / "classroom.mp4"

DATA_DIR = BACKEND / "data"
FACES_DIR = DATA_DIR / "faces"
STUDENTS_CSV = DATA_DIR / "students.csv"
TEACHERS_CSV = DATA_DIR / "teachers.csv"

STUDENT_COLS = ["student_id", "name", "class", "seat", "registered_at", "photo"]
TEACHER_COLS = ["name", "subject", "class", "registered_at"]

BEHAVIOURS = ["Attentive", "Distracted", "Sleepy", "Phone"]

st.set_page_config(
    page_title="Classroom Attention Monitor",
    page_icon="🎓",
    layout="wide",
)


# --- helpers ---------------------------------------------------------------

def find_module(stem):
    """Import Backend/**/<stem>.py as a package module, wherever it lives."""

    for path in sorted(BACKEND.rglob(f"{stem}.py")):

        if "venv" in path.parts or "__pycache__" in path.parts:
            continue

        dotted = ".".join(path.relative_to(ROOT).with_suffix("").parts)

        try:
            return importlib.import_module(dotted), path, None
        except Exception as error:  # report it in the UI
            return None, path, f"{type(error).__name__}: {error}"

    return None, None, f"{stem}.py not found under {BACKEND}"


def spectral_quality(signal, fps):
    """
    Estimate signal quality (0-100) and heart rate (BPM) from the spectrum.

    Quality = share of band power (0.7-3.0 Hz) close to the strongest peak.
    """

    x = np.asarray(signal, dtype=float)

    if len(x) < 30 or not fps:
        return 0.0, 0.0

    x = x - np.mean(x)
    n = len(x)

    freqs = np.fft.rfftfreq(n, 1.0 / fps)
    power = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2

    band = (freqs >= 0.7) & (freqs <= 3.0)

    if not band.any() or power[band].sum() == 0:
        return 0.0, 0.0

    peak_freq = freqs[band][np.argmax(power[band])]
    near_peak = band & (np.abs(freqs - peak_freq) <= 0.1)

    quality = 100.0 * power[near_peak].sum() / power[band].sum()

    return float(min(100.0, quality)), float(peak_freq * 60.0)


def build_heatmap(reference_rgb, points, sigma=40, alpha=0.65):
    """Overlay a density heat map of (x, y) points on a reference frame."""

    h, w = reference_rgb.shape[:2]

    grid = np.zeros((h, w), dtype=np.float32)

    for cx, cy in points:
        if 0 <= cx < w and 0 <= cy < h:
            grid[int(cy), int(cx)] += 1.0

    grid = cv2.GaussianBlur(grid, (0, 0), sigma)

    if grid.max() <= 0:
        return reference_rgb

    norm = (grid / grid.max() * 255).astype(np.uint8)

    colour = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)

    mask = (norm.astype(np.float32) / 255.0)[..., None] * alpha

    blended = reference_rgb * (1 - mask) + colour * mask

    return blended.astype(np.uint8)


def classify_behaviour(class_name):
    """Map a YOLO class name to one of BEHAVIOURS (or None)."""

    name = str(class_name).lower()

    if "distract" in name:
        return "Distracted"
    if "attent" in name:
        return "Attentive"
    if "sleep" in name:
        return "Sleepy"
    if "phone" in name:
        return "Phone"

    return None


def make_row(frame_idx, time_s, counts):

    total = sum(counts.values())

    row = {"frame": int(frame_idx), "time_s": round(float(time_s), 1)}

    for b in BEHAVIOURS:
        row[b.lower()] = int(counts[b])

    row["attention_pct"] = (
        counts["Attentive"] / total * 100 if total else np.nan
    )

    return row


def make_result(rows, points, reference, source, **extra):

    result = {
        "timeline": pd.DataFrame(rows),
        "points": points,
        "reference": reference,
        "source": source,
    }

    result.update(extra)

    return result


def summarize(result):
    """Per-behaviour averages/totals and overall attention %."""

    if result is None or len(result["timeline"]) == 0:
        return None

    tl = result["timeline"]

    totals = {b: int(tl[b.lower()].sum()) for b in BEHAVIOURS}
    averages = {b: float(tl[b.lower()].mean()) for b in BEHAVIOURS}

    seen = sum(totals.values())

    return {
        "totals": totals,
        "averages": averages,
        "attention_pct": (totals["Attentive"] / seen * 100) if seen else None,
        "frames": len(tl),
    }


def insights(summary):
    """Simple rule-based teaching suggestions."""

    if summary is None:
        return [("info", "Run an analysis in the Monitoring tab to see insights.")]

    out = []

    pct = summary["attention_pct"]
    avg = summary["averages"]

    if pct is None:
        return [("info", "No students were detected in the analyzed frames.")]

    if pct < 40:
        out.append(("error", "Low attention overall. Consider changing the activity or asking a question to re-engage the class."))
    elif pct < 70:
        out.append(("warning", "Moderate attention. A short interactive task could help."))
    else:
        out.append(("success", "The class looks engaged."))

    if avg["Sleepy"] > 0:
        out.append(("warning", f"Sleepy students detected (about {avg['Sleepy']:.1f} per frame). A short break or movement activity may help."))

    if avg["Phone"] > 0:
        out.append(("warning", f"Phone use detected (about {avg['Phone']:.1f} per frame)."))

    seen = sum(avg.values())

    if seen and avg["Distracted"] / seen > 0.3:
        out.append(("warning", "More than 30% of detected students look distracted. Check the heat map for where."))

    return out


def load_table(path, columns):

    if path.exists():
        df = pd.read_csv(path, dtype=str).fillna("")
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        return df[columns]

    return pd.DataFrame(columns=columns)


def save_table(df, path):

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_-]", "_", text.strip())


# --- model and cached heavy work -------------------------------------------

@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model():
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


def detect_behaviours(model, frame_bgr, conf, annotate=False):
    """Run YOLO on one frame -> (counts, points, annotated RGB or None)."""

    result = model.predict(frame_bgr, conf=conf, verbose=False, save=False)[0]

    counts = {b: 0 for b in BEHAVIOURS}
    points = {b: [] for b in BEHAVIOURS}

    if result.boxes is not None and len(result.boxes) > 0:

        for box, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):

            behaviour = classify_behaviour(result.names[int(cls)])

            if behaviour is None:
                continue

            counts[behaviour] += 1
            points[behaviour].append(
                ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            )

    annotated = None

    if annotate:
        annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

    return counts, points, annotated


@st.cache_data(show_spinner=False)
def analyze_attention(video_path, file_stamp, conf, n_samples):
    """Run YOLO on evenly spaced video frames."""

    model = load_model()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    indices = np.linspace(
        0, max(total - 1, 0), min(n_samples, max(total, 1))
    ).astype(int)

    rows = []
    points = {b: [] for b in BEHAVIOURS}
    reference = None

    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        if reference is None:
            reference = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        counts, pts, _ = detect_behaviours(model, frame, conf)

        for b in BEHAVIOURS:
            points[b].extend(pts[b])

        rows.append(make_row(idx, idx / fps, counts))

    cap.release()

    return make_result(
        rows,
        points,
        reference,
        f"Video: {Path(video_path).name}",
        source_type="video",
        video_path=video_path,
        file_stamp=file_stamp,
        fps=float(fps),
        total_frames=total,
    )


@st.cache_data(show_spinner=False)
def annotate_frame(video_path, file_stamp, frame_idx, conf):
    """Return one video frame with YOLO boxes drawn (RGB)."""

    model = load_model()

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    return detect_behaviours(model, frame, conf, annotate=True)[2]


@st.cache_data(show_spinner=False, max_entries=20)
def analyze_image_bytes(data, conf, label):
    """Analyze one still image (upload or camera snapshot)."""

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not read that image.")

    model = load_model()

    counts, points, annotated = detect_behaviours(
        model, frame, conf, annotate=True
    )

    return make_result(
        [make_row(0, 0, counts)],
        points,
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        label,
        source_type="image",
        annotated=annotated,
    )


def file_stamp(path):
    """Cache key that changes when the file changes."""

    p = Path(path)

    if not p.exists():
        return "missing"

    return f"{p.stat().st_size}-{int(p.stat().st_mtime)}"


@st.cache_data(show_spinner=False)
def run_rppg(video_path, stamp, detect_every, detect_scale, module_stamp):
    """
    Run the rPPG pipeline once and keep the result.

    Works with both the new rppg.py (returns a dict and accepts the speed
    settings) and an older one (returns nothing and saves .npy files).
    """

    module, _, error = find_module("rppg")

    if module is None:
        raise ImportError(error)

    module = importlib.reload(module)  # pick up an edited/replaced file

    params = inspect.signature(module.process_video).parameters

    kwargs = {
        key: value
        for key, value in (
            ("detect_every", detect_every),
            ("detect_scale", detect_scale),
        )
        if key in params
    }

    out = module.process_video(video_path, **kwargs)

    legacy = not isinstance(out, dict)

    if legacy:

        results_dir = Path(getattr(module, "RESULTS_DIR", BACKEND / "results"))

        signal_file = results_dir / "green_signal.npy"
        fps_file = results_dir / "video_fps.npy"

        if not signal_file.exists() or not fps_file.exists():
            raise RuntimeError(
                "This rppg.py returned no signal and saved no files in "
                f"{results_dir}."
            )

        green = [float(v) for v in np.load(signal_file)]

        out = {
            "green": green,
            "fps": float(np.load(fps_file)[0]),
            "frames": len(green),
            "preview": [],
        }

    return {
        "green": out["green"],
        "fps": out["fps"],
        "frames": out["frames"],
        "preview": out["preview"],
        "legacy": legacy,
    }


# --- display helpers (cards, gauge, distribution) ---------------------------

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

BEHAVIOUR_COLOURS = {
    "Attentive": "#2fbf8a",
    "Distracted": "#f5a524",
    "Sleepy": "#8b6cf0",
    "Phone": "#ff4d4f",
}


def card_html(label, value, caption):
    return (
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>'
        f'<div class="card-cap">{caption}</div></div>'
    )


def show_plot(fig):
    """st.plotly_chart across Streamlit versions."""

    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_gauge(value):

    st.markdown("**Classroom performance**")

    if not HAS_PLOTLY:
        st.progress(int(max(0, min(100, value))) / 100.0)
        st.caption(f"{value:.0f} / 100")
        return

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 56}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [20, 40, 60, 80]},
            "bar": {"color": "#1a1af0"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#4a2347"},
                {"range": [40, 60], "color": "#7a4b2a"},
                {"range": [60, 100], "color": "#2f6f7a"},
            ],
        },
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e8e8ff"},
    )

    show_plot(fig)


def show_distribution(summary):

    st.markdown("**Classroom behaviour distribution**")

    values = [
        0 if summary is None else round(summary["averages"][b], 2)
        for b in BEHAVIOURS
    ]

    if not HAS_PLOTLY:
        st.bar_chart(
            pd.DataFrame({"Behaviour": BEHAVIOURS, "Count": values})
            .set_index("Behaviour")
        )
        return

    fig = go.Figure(go.Bar(
        x=BEHAVIOURS,
        y=values,
        text=values,
        textposition="outside",
        marker_color=[BEHAVIOUR_COLOURS[b] for b in BEHAVIOURS],
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e8e8ff"},
        xaxis_title="Behavior",
        yaxis_title="Average count per frame",
    )

    show_plot(fig)


def run_live(source, label, ph_img, ph_info, conf):
    """
    Live detection loop for a webcam index or a stream URL.
    Runs until the user switches the toggle off (Streamlit reruns).
    """

    if not MODEL_PATH.exists():
        ph_info.error(f"Model not found: {MODEL_PATH}")
        return

    model = load_model()

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        ph_info.error(
            f"Could not open {label}. For a webcam, allow camera access for "
            "your terminal in System Settings > Privacy & Security > Camera. "
            "For CCTV, check the stream URL and network."
        )
        return

    rows = []
    points = {b: [] for b in BEHAVIOURS}
    reference = None
    started = time.time()
    n = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok or frame is None:
                ph_info.warning("The stream ended or no frame was received.")
                break

            counts, pts, annotated = detect_behaviours(
                model, frame, conf, annotate=True
            )

            if reference is None:
                reference = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            n += 1

            rows.append(make_row(n, time.time() - started, counts))
            rows = rows[-300:]

            for b in BEHAVIOURS:
                points[b].extend(pts[b])
                points[b] = points[b][-5000:]

            st.session_state["attention"] = make_result(
                rows,
                points,
                reference,
                f"Live: {label}",
                source_type="live",
                annotated=annotated,
            )

            row = rows[-1]

            ph_img.image(annotated, width="stretch")

            pct = row["attention_pct"]

            ph_info.caption(
                f"Attentive {row['attentive']} · Distracted {row['distracted']} "
                f"· Sleepy {row['sleepy']} · Phone {row['phone']} · "
                + ("Attention n/a" if pd.isna(pct) else f"Attention {pct:.0f}%")
            )

            time.sleep(0.03)

    finally:
        cap.release()


# ==================================================
# SIDEBAR (settings only)
# ==================================================

state = st.session_state

st.sidebar.title("🎓 Attention Monitor")

conf = st.sidebar.slider("YOLO confidence", 0.10, 0.90, 0.35, 0.05)
n_samples = st.sidebar.slider("Frames to analyze (video)", 10, 120, 40, 10)

with st.sidebar.expander("rPPG speed settings"):
    detect_every = st.slider("Detect face every N frames", 1, 10, 3)
    detect_scale = st.select_slider(
        "Detection image scale", options=[0.25, 0.5, 0.75, 1.0], value=0.5
    )

st.sidebar.caption("Upload videos and pick a source in the Monitoring tab.")

# ==================================================
# STYLE
# ==================================================

st.markdown(
    """
<style>
.hero {
    background: linear-gradient(135deg, #2a0fd6 0%, #3b5bf0 55%, #3a9bf5 100%);
    border-radius: 18px; padding: 26px 30px; color: #fff;
}
.hero-title {font-size: 2rem; font-weight: 700; line-height: 1.2;}
.hero-sub {margin-top: 10px; opacity: 0.92; font-size: 0.95rem;}
.alert {border-radius: 16px; padding: 18px 24px; color: #fff; margin: 14px 0;}
.alert-title {font-size: 1.15rem; font-weight: 700;}
.alert-sub {opacity: 0.9; font-size: 0.85rem; margin-top: 4px;}
.alert-red {background: linear-gradient(90deg, #c92a5a, #ff4d4f);}
.alert-amber {background: linear-gradient(90deg, #b8741a, #f5a524);}
.alert-green {background: linear-gradient(90deg, #1f8a5b, #2fbf8a);}
.alert-grey {background: linear-gradient(90deg, #3a3f5c, #505777);}
.card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px; padding: 16px 18px; min-height: 118px;
}
.card-label {font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.8;}
.card-value {font-size: 2.1rem; font-weight: 700; margin: 6px 0 2px 0;}
.card-cap {font-size: 0.78rem; color: #9aa4ff;}
</style>
""",
    unsafe_allow_html=True,
)

# ==================================================
# HEADER
# ==================================================

head_left, head_right = st.columns([6, 1])

head_left.markdown(
    '<div class="hero"><div class="hero-title">🎓 AI Classroom Monitoring</div>'
    '<div class="hero-sub">Attention, alertness and engagement analytics '
    "in one place</div></div>",
    unsafe_allow_html=True,
)

if head_right.button("🔄 Refresh"):
    st.cache_data.clear()
    state.clear()
    st.rerun()

(
    tab_dash, tab_monitor, tab_students, tab_teacher, tab_reports,
    tab_register,
) = st.tabs([
    "📊 Dashboard", "👥 Monitoring", "🧑‍🎓 Students", "🧑‍🏫 Teacher",
    "📄 Reports", "➕ Register",
])

live_request = None

# --------------------------------------------------
# MONITORING (runs first so the other tabs see fresh results)
# --------------------------------------------------

with tab_monitor:

    st.subheader("Input source")

    source = st.radio(
        "Source",
        [
            "Video file",
            "Upload image",
            "Camera snapshot",
            "Webcam (live)",
            "CCTV / IP camera (live)",
        ],
        horizontal=True,
        key="source_choice",
    )

    if source == "Video file":

        video_upload = st.file_uploader(
            "Upload a classroom video (or leave empty to use the default)",
            type=["mp4", "mov", "avi", "mkv"],
            key="video_upload",
        )

        if video_upload is not None:

            saved = Path(tempfile.gettempdir()) / video_upload.name

            if not saved.exists() or saved.stat().st_size != video_upload.size:
                saved.write_bytes(video_upload.getbuffer())

            state["video_path"] = str(saved)

        else:
            state.pop("video_path", None)

        video_path = state.get("video_path") or str(DEFAULT_VIDEO)

        st.caption(
            f"Using: **{Path(video_path).name}**"
            + ("" if video_upload is not None else " (default video)")
        )

        btn_a, btn_b = st.columns(2)

        analyze_clicked = btn_a.button("▶ Analyze video", key="btn_video")
        heart_clicked = btn_b.button("❤ Run rPPG analysis", key="btn_rppg")

        if analyze_clicked:

            if not Path(video_path).exists():
                st.error(f"Video not found: {video_path}")

            elif not MODEL_PATH.exists():
                st.error(f"Model not found: {MODEL_PATH}")

            else:
                with st.spinner("Running YOLO on sampled frames..."):
                    try:
                        state["attention"] = analyze_attention(
                            video_path, file_stamp(video_path), conf, n_samples
                        )
                    except Exception as error:
                        st.error(f"{type(error).__name__}: {error}")

        if heart_clicked:

            if not Path(video_path).exists():
                state["rppg"] = None
                state["rppg_error"] = f"Video not found: {video_path}"

            else:
                _, rppg_path, _ = find_module("rppg")

                with st.spinner("Extracting pulse signal from the video..."):
                    try:
                        state["rppg"] = run_rppg(
                            video_path,
                            file_stamp(video_path),
                            detect_every,
                            detect_scale,
                            int(rppg_path.stat().st_mtime) if rppg_path else 0,
                        )
                        state["rppg_error"] = None
                    except Exception as error:
                        state["rppg"] = None
                        state["rppg_error"] = f"{type(error).__name__}: {error}"

        if state.get("rppg_error"):
            st.error(state["rppg_error"])
        elif state.get("rppg") is not None:
            st.success("rPPG finished. See the Dashboard tab.")
            if state["rppg"].get("legacy"):
                st.warning(
                    "Your rppg.py is an older version (no speed settings), so "
                    "it read the saved signal files instead. Replace it with "
                    "the latest rppg.py for faster, more reliable results."
                )

        if Path(video_path).exists():
            with st.expander("Preview video"):
                st.video(video_path)

    elif source in ("Upload image", "Camera snapshot"):

        if source == "Upload image":
            picture = st.file_uploader(
                "Classroom image",
                type=["jpg", "jpeg", "png", "webp"],
                key="image_upload",
            )
            label = "Image: upload"
        else:
            picture = st.camera_input(
                "Take a classroom snapshot", key="camera_snapshot"
            )
            label = "Image: camera snapshot"

        if picture is not None:

            if not MODEL_PATH.exists():
                st.error(f"Model not found: {MODEL_PATH}")

            else:
                with st.spinner("Detecting behaviours..."):
                    try:
                        result = analyze_image_bytes(
                            picture.getvalue(), conf, label
                        )
                        state["attention"] = result
                        st.image(result["annotated"], width="stretch")
                    except Exception as error:
                        st.error(f"{type(error).__name__}: {error}")

    elif source == "Webcam (live)":

        st.caption(
            "Uses the camera on the computer running Streamlit. Detection "
            "runs until you switch it off."
        )

        cam_index = st.number_input("Camera index", 0, 10, 0, key="cam_index")
        live_on = st.toggle("Start live detection", key="live_webcam")

        ph_img = st.empty()
        ph_info = st.empty()

        if live_on:
            live_request = (int(cam_index), "webcam", ph_img, ph_info)

    else:  # CCTV / IP camera

        st.caption(
            "Enter the camera's stream address (for example rtsp://... or "
            "http://...). It is hidden while you type and is not saved."
        )

        stream_url = st.text_input(
            "Stream URL", type="password", key="cctv_url"
        )

        live_on = st.toggle("Start live detection", key="live_cctv")

        ph_img = st.empty()
        ph_info = st.empty()

        if live_on and stream_url:
            live_request = (stream_url, "CCTV stream", ph_img, ph_info)
        elif live_on:
            st.warning("Enter the stream URL first.")

    # ---------------- results of the latest analysis ----------------

    attention = state.get("attention")

    if attention is not None and len(attention["timeline"]) > 0:

        tl = attention["timeline"]

        st.divider()
        st.subheader("Attention results")
        st.caption(f"Source: {attention['source']}")

        if len(tl) > 1:

            st.line_chart(
                tl.set_index("time_s")[["attention_pct"]],
                y_label="Attentive students (%)",
                x_label="Time (s)",
            )

            st.bar_chart(
                tl.set_index("time_s")[[b.lower() for b in BEHAVIOURS]],
                x_label="Time (s)",
                y_label="Students",
            )

        st.markdown("**Frame viewer**")

        if attention.get("source_type") == "video":

            chosen = st.select_slider(
                "Frame",
                options=list(tl["frame"]),
                format_func=lambda f: (
                    f"{tl.loc[tl['frame'] == f, 'time_s'].iloc[0]} s"
                ),
                key="frame_pick",
            )

            frame_image = annotate_frame(
                attention["video_path"],
                attention["file_stamp"],
                int(chosen),
                conf,
            )

        else:
            frame_image = attention.get("annotated")

        if frame_image is not None:
            st.image(frame_image, width="stretch")

        st.divider()
        st.subheader("Heat map")

        which = st.radio("Show", BEHAVIOURS, horizontal=True, key="heat_which")
        spread = st.slider("Smoothing", 10, 120, 40, 5, key="heat_sigma")

        points = attention["points"][which]

        if attention["reference"] is None or len(points) == 0:
            st.warning(f"No {which.lower()} detections to plot.")
        else:
            st.image(
                build_heatmap(attention["reference"], points, sigma=spread),
                caption=(
                    f"{which} density over {len(tl)} frame(s) "
                    "(fixed camera assumed)"
                ),
                width="stretch",
            )

        with st.expander("Data table"):
            st.dataframe(tl, width="stretch")

# --------------------------------------------------
# HEART RATE RESULT (after Monitoring has processed its buttons)
# --------------------------------------------------

rppg_out = state.get("rppg")

heart = None

if rppg_out and len(rppg_out["green"]) >= 30:

    signal_module, _, _ = find_module("signal_processing")

    quality, spectral_bpm = spectral_quality(
        rppg_out["green"], rppg_out["fps"]
    )

    face_visibility = len(rppg_out["green"]) / max(rppg_out["frames"], 1) * 100

    if signal_module is not None:

        heart = signal_module.analyze_signal(
            rppg_out["green"],
            rppg_out["fps"],
            signal_quality=quality,
            face_visibility=min(face_visibility, 100.0),
            save=False,
        )

        if heart is not None:
            heart["fig"] = signal_module.plot_waveform(heart)
            heart["spectral_bpm"] = spectral_bpm
            heart["quality"] = quality
            heart["face_visibility"] = min(face_visibility, 100.0)

# --------------------------------------------------
# REGISTER (before Students, so the list is fresh)
# --------------------------------------------------

with tab_register:

    st.subheader("Register a student")

    with st.form("student_form", clear_on_submit=True):

        name = st.text_input("Full name")
        student_id = st.text_input("Student ID")
        klass = st.text_input("Class")
        seat = st.text_input("Seat (optional, e.g. row 2, seat 4)")

        photo_file = st.file_uploader(
            "Face photo (optional)", type=["jpg", "jpeg", "png"],
            key="reg_photo_upload",
        )
        photo_cam = st.camera_input(
            "...or take a photo (optional)", key="reg_photo_camera"
        )

        consent = st.checkbox(
            "I have consent to store this student's photo"
        )

        submitted = st.form_submit_button("Register student")

    if submitted:

        students = load_table(STUDENTS_CSV, STUDENT_COLS)
        photo = photo_file or photo_cam
        clean_id = safe_name(student_id)

        if not name.strip() or not clean_id:
            st.error("Name and Student ID are required.")

        elif clean_id in set(students["student_id"]):
            st.error(f"Student ID '{clean_id}' is already registered.")

        elif photo is not None and not consent:
            st.error("Tick the consent box to store a photo, or remove the photo.")

        else:

            photo_name = ""

            if photo is not None:
                FACES_DIR.mkdir(parents=True, exist_ok=True)
                photo_name = f"{clean_id}.jpg"
                (FACES_DIR / photo_name).write_bytes(photo.getvalue())

            students.loc[len(students)] = [
                clean_id,
                name.strip(),
                klass.strip(),
                seat.strip(),
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                photo_name,
            ]

            save_table(students, STUDENTS_CSV)

            st.success(f"Registered {name.strip()} ({clean_id}).")

# --------------------------------------------------
# STUDENTS
# --------------------------------------------------

with tab_students:

    students = load_table(STUDENTS_CSV, STUDENT_COLS)

    st.metric("Registered students", len(students))

    if len(students) == 0:
        st.info("No students yet. Use the Register tab to add some.")

    else:

        query = st.text_input("Search by name, ID or class", key="student_search")

        shown = students

        if query.strip():
            q = query.strip().lower()
            shown = students[
                students.apply(
                    lambda r: q in " ".join(r.astype(str)).lower(), axis=1
                )
            ]

        st.dataframe(shown.drop(columns=["photo"]), width="stretch")

        st.subheader("Student details")

        pick = st.selectbox(
            "Student",
            list(students["student_id"]),
            format_func=lambda sid: (
                f"{sid} - {students.loc[students['student_id'] == sid, 'name'].iloc[0]}"
            ),
            key="student_pick",
        )

        photo_name = students.loc[
            students["student_id"] == pick, "photo"
        ].iloc[0]

        if photo_name and (FACES_DIR / photo_name).exists():
            st.image(str(FACES_DIR / photo_name), width=180)
        else:
            st.caption("No photo stored.")

        if st.button("Remove this student", key="remove_student"):

            if photo_name and (FACES_DIR / photo_name).exists():
                (FACES_DIR / photo_name).unlink()

            save_table(students[students["student_id"] != pick], STUDENTS_CSV)

            st.rerun()

# --------------------------------------------------
# TEACHER
# --------------------------------------------------

with tab_teacher:

    st.subheader("Teacher profile")

    with st.form("teacher_form", clear_on_submit=True):

        t_name = st.text_input("Teacher name")
        t_subject = st.text_input("Subject")
        t_class = st.text_input("Class taught")

        t_submit = st.form_submit_button("Save teacher")

    if t_submit:

        if not t_name.strip():
            st.error("Teacher name is required.")

        else:

            teachers = load_table(TEACHERS_CSV, TEACHER_COLS)

            teachers.loc[len(teachers)] = [
                t_name.strip(),
                t_subject.strip(),
                t_class.strip(),
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ]

            save_table(teachers, TEACHERS_CSV)

            st.success(f"Saved {t_name.strip()}.")

    teachers = load_table(TEACHERS_CSV, TEACHER_COLS)

    if len(teachers) > 0:
        st.dataframe(teachers, width="stretch")

    st.subheader("Classroom insights")

    st.caption("Simple rules based on the latest analysis, not a diagnosis.")

    for kind, message in insights(summarize(state.get("attention"))):
        getattr(st, kind)(message)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

with tab_dash:

    attention = state.get("attention")
    summary = summarize(attention)

    pct = summary["attention_pct"] if summary else None

    # Fuse YOLO attention with rPPG alertness when both exist
    if pct is None:
        fused = None
        fused_note = "No data yet"
    elif heart is not None:
        fused = 0.7 * pct + 0.3 * heart["alertness_score"]
        fused_note = "YOLO + rPPG fused"
    else:
        fused = pct
        fused_note = "YOLO only (no rPPG yet)"

    if pct is None:
        banner = ("alert-grey", "No analysis yet", "Open Monitoring and choose a source")
    elif pct < 40:
        banner = ("alert-red", "Immediate intervention required", "Low Attention")
    elif pct < 70:
        banner = ("alert-amber", "Attention needs a boost", "Moderate Attention")
    else:
        banner = ("alert-green", "Class is engaged", "High Attention")

    st.markdown(
        f'<div class="alert {banner[0]}"><div class="alert-title">{banner[1]}'
        f'</div><div class="alert-sub">{banner[2]}</div></div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.markdown(
        card_html(
            "Attention",
            "N/A" if pct is None else f"{pct:.1f}%",
            "From YOLO detections",
        ),
        unsafe_allow_html=True,
    )
    k2.markdown(
        card_html(
            "Alertness",
            "N/A" if fused is None else f"{fused:.1f}%",
            fused_note,
        ),
        unsafe_allow_html=True,
    )
    k3.markdown(
        card_html(
            "Heart rate",
            "N/A" if heart is None else f"{heart['bpm']:.0f} BPM",
            "rPPG estimate (experimental)",
        ),
        unsafe_allow_html=True,
    )
    k4.markdown(
        card_html(
            "rPPG alertness",
            "N/A" if heart is None else f"{heart['alertness_score']:.0f}",
            "Physiological" if heart is None else heart["status"],
        ),
        unsafe_allow_html=True,
    )

    st.write("")

    g_col, b_col = st.columns(2)

    with g_col:
        show_gauge(0 if fused is None else fused)

    with b_col:
        show_distribution(summary)

    with st.expander("rPPG pulse waveform", expanded=heart is not None):

        st.caption(
            "Experimental: pulse estimated from forehead colour changes. "
            "Distance, lighting and head movement limit accuracy, so treat "
            "it as a rough signal, not a measure of attention."
        )

        if state.get("rppg_error"):
            st.error(state["rppg_error"])

        elif rppg_out is None:
            st.info(
                "Go to Monitoring, choose Video file and click "
                "**Run rPPG analysis**."
            )

        elif heart is None:
            st.warning(
                f"Only {len(rppg_out['green'])} signal samples were collected "
                "(not enough). The face may be too small or not detected. Try "
                "'Detection image scale' = 1.0 in the sidebar."
            )

        else:

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("BPM (peak count)", f"{heart['bpm']:.0f}")
            m2.metric("BPM (spectrum)", f"{heart['spectral_bpm']:.0f}")
            m3.metric("Signal quality", f"{heart['quality']:.0f}%")
            m4.metric("Face visible", f"{heart['face_visibility']:.0f}%")

            if abs(heart["bpm"] - heart["spectral_bpm"]) > 15 or heart["quality"] < 30:
                st.warning(
                    "The two BPM estimates disagree or the signal is weak, so "
                    "this reading is unreliable."
                )

            st.pyplot(heart["fig"])

            if rppg_out["preview"]:

                cols = st.columns(min(3, len(rppg_out["preview"])))

                for i, img in enumerate(rppg_out["preview"][:3]):
                    cols[i].image(img, width="stretch")

    if summary is not None:
        st.caption(f"Latest analysis: {attention['source']}")

    with st.expander("Modules in use"):

        for stem in ("rppg", "signal_processing", "face_detector", "alertness"):

            module, path, error = find_module(stem)

            if error:
                st.write(f"❌ `{stem}`: {error}")
            else:
                st.write(f"✅ `{stem}` → `{path.relative_to(ROOT)}`")

# --------------------------------------------------
# REPORTS
# --------------------------------------------------

with tab_reports:

    attention = state.get("attention")
    summary = summarize(attention)

    if summary is None:
        st.info("Analyze a source in the Monitoring tab to build a report.")

    else:

        st.subheader("Session summary")

        st.caption(
            f"Source: {attention['source']} · generated "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        summary_df = pd.DataFrame({
            "Behaviour": BEHAVIOURS,
            "Average per frame": [round(summary["averages"][b], 2) for b in BEHAVIOURS],
            "Total detections": [summary["totals"][b] for b in BEHAVIOURS],
        })

        st.dataframe(summary_df, width="stretch")

        st.metric(
            "Overall attention",
            "n/a" if summary["attention_pct"] is None
            else f"{summary['attention_pct']:.0f}%",
        )

        for kind, message in insights(summary):
            getattr(st, kind)(message)

        if heart is not None:
            st.write(
                f"Heart rate (experimental): {heart['bpm']:.0f} BPM peak count, "
                f"{heart['spectral_bpm']:.0f} BPM spectrum, signal quality "
                f"{heart['quality']:.0f}%."
            )

        st.download_button(
            "Download summary (CSV)",
            data=summary_df.to_csv(index=False),
            file_name="attention_summary.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download frame-by-frame data (CSV)",
            data=attention["timeline"].to_csv(index=False),
            file_name="attention_timeline.csv",
            mime="text/csv",
        )

# ==================================================
# LIVE LOOP (last, so every tab above has already been drawn)
# ==================================================

if live_request is not None:

    live_source, live_label, live_img, live_info = live_request

    run_live(live_source, live_label, live_img, live_info, conf)
