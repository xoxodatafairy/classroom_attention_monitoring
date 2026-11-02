from pathlib import Path
import json
import shutil
import cv2


import numpy as np

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form
)

from Backend.detect import analyze_image
from threading import Thread

from Backend.services.live_detector import (
    run_live_detection
)

# ==================================================
# FASTAPI
# ==================================================
live_thread = None
app = FastAPI(
    title="AI Classroom Monitoring API"
)

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

UPLOAD_IMAGE_DIR = (
    BASE_DIR /
    "uploads" /
    "images"
)

UPLOAD_VIDEO_DIR = (
    BASE_DIR /
    "uploads" /
    "videos"
)

UPLOAD_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

UPLOAD_VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==================================================
# HELPERS
# ==================================================

def safe_filename(filename):

    return Path(filename).name


def load_score(filename):

    try:

        filepath = (
            RESULTS_DIR /
            filename
        )

        if not filepath.exists():

            return 0.0

        return float(
            np.load(filepath)[0]
        )

    except Exception:

        return 0.0


# ==================================================
# ROOT
# ==================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "project": "AI Classroom Monitoring"
    }


# ==================================================
# INDIVIDUAL SCORES
# ==================================================

@app.get("/attention")
def attention():

    return {
        "attention_score":
        load_score(
            "attention_score.npy"
        )
    }


@app.get("/rppg")
def rppg():

    return {
        "alertness_score":
        load_score(
            "alertness_score.npy"
        )
    }


@app.get("/score")
def score():

    return {
        "final_score":
        load_score(
            "final_score.npy"
        )
    }


# ==================================================
# DASHBOARD
# ==================================================

@app.get("/dashboard")
def dashboard():

    attention_score = load_score(
        "attention_score.npy"
    )

    alertness_score = load_score(
        "alertness_score.npy"
    )

    final_score = load_score(
        "final_score.npy"
    )

    if final_score >= 80:

        status = "Highly Attentive ✅"

    elif final_score >= 60:

        status = "Moderately Attentive ⚠️"

    else:

        status = "Low Attention ❌"

    return {

        "attention_score":
        attention_score,

        "alertness_score":
        alertness_score,

        "final_score":
        final_score,

        "status":
        status
    }


# ==================================================
# ANALYTICS
# ==================================================

@app.get("/analytics")
def analytics():

    analytics_file = (
        RESULTS_DIR /
        "analytics.json"
    )

    try:

        with open(
            analytics_file,
            "r"
        ) as f:

            data = json.load(f)

        return {

            "attentive_students":
            data.get(
                "Attentive",
                0
            ),

            "distracted_students":
            data.get(
                "Distracted",
                0
            ),

            "sleepy_students":
            data.get(
                "Sleepy",
                0
            ),

            "phone_usage":
            data.get(
                "Phone",
                0
            )
        }

    except Exception:

        return {

            "attentive_students": 0,
            "distracted_students": 0,
            "sleepy_students": 0,
            "phone_usage": 0
        }


# ==================================================
# RAW DETECTIONS
# ==================================================

@app.get("/detections")
def detections():

    analytics_file = (
        RESULTS_DIR /
        "analytics.json"
    )

    try:

        with open(
            analytics_file,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


# ==================================================
# IMAGE UPLOAD
# ==================================================

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...)
):

    filename = safe_filename(
        file.filename
    )

    filepath = (
        UPLOAD_IMAGE_DIR /
        filename
    )

    with open(
        filepath,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    return {

        "status":
        "success",

        "file":
        str(filepath)
    }


# ==================================================
# ANALYZE IMAGE
# ==================================================

@app.post("/analyze-image")
async def analyze_image_endpoint(
    file: UploadFile = File(...)
):

    try:

        filename = Path(
            file.filename
        ).name

        filepath = (
            UPLOAD_IMAGE_DIR /
            filename
        )

        with open(
            filepath,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        result = analyze_image(
            filepath
        )

        return {
            "status": "success",
            "result": result
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }
# ==================================================
# VIDEO UPLOAD
# ==================================================

@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...)
):

    filename = safe_filename(
        file.filename
    )

    filepath = (
        UPLOAD_VIDEO_DIR /
        filename
    )

    with open(
        filepath,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    return {

        "status":
        "success",

        "file":
        str(filepath)
    }


# ==================================================
# WEBCAM CHECK
# ==================================================

@app.get("/start-webcam")
def start_webcam():

    camera = None

    try:

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():

            return {
                "status": "error",
                "message": "Webcam unavailable"
            }

        return {
            "status": "success",
            "message": "Webcam ready"
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

    finally:

        if camera is not None:

            camera.release()


# ==================================================
# TEACHER ALERT
# ==================================================

@app.get("/teacher-alert")
def teacher_alert():

    final_score = load_score(
        "final_score.npy"
    )

    if final_score >= 80:

        return {

            "score":
            final_score,

            "status":
            "Highly Attentive",

            "alert_color":
            "GREEN"
        }

    elif final_score >= 60:

        return {

            "score":
            final_score,

            "status":
            "Attention Dropping",

            "alert_color":
            "YELLOW"
        }

    return {

        "score":
        final_score,

        "status":
        "Intervention Required",

        "alert_color":
        "RED"
    }
    
@app.get("/detections")
def detections():

    analytics_file = (
        RESULTS_DIR /
        "analytics.json"
    )

    try:

        with open(
            analytics_file,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {}
@app.post("/start-monitoring")
def start_monitoring(
    source: str
):

    global live_thread

    if live_thread and live_thread.is_alive():

        return {
            "status": "already_running"
        }

    live_thread = Thread(

        target=run_live_detection,

        args=(source,)
    )

    live_thread.daemon = True

    live_thread.start()

    return {
        "status": "started"
    }
    # ==================================================
# STUDENT METRICS
# ==================================================

@app.get("/students")
def students():

    file_path = (
        RESULTS_DIR /
        "student_metrics.json"
    )

    try:

        with open(
            file_path,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {}
    
@app.get("/students")
def students():

    file_path = (
        RESULTS_DIR /
        "student_metrics.json"
    )

    try:

        with open(
            file_path,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {}
    
    
@app.get("/attention-history")
def attention_history():

    file_path = (
        RESULTS_DIR /
        "attention_history.json"
    )

    try:

        with open(file_path, "r") as f:
            return json.load(f)

    except Exception:

        return []
@app.get("//attention-history")
def history():

    file_path = (
        RESULTS_DIR /
        "attention_history.json"
    )

    try:

        with open(
            file_path,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {

            "labels": [],

            "attention": [],

            "engagement": [],

            "pulse": []
        }
        
        
        
        
@app.post("/register-student")
async def register_student(
    student_id: str = Form(...),
    student_name: str = Form(...),
    file: UploadFile = File(...)
):

    student_dir = (
        BASE_DIR /
        "student_registry" /
        student_id
    )

    student_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = (
        student_dir /
        "profile.jpg"
    )

    with open(
        image_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    metadata = {

        "student_id": student_id,

        "student_name": student_name,

        "image": "profile.jpg"
    }

    with open(
        student_dir /
        "student.json",
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    return {
        "status": "success",
        "student_id": student_id
    }
