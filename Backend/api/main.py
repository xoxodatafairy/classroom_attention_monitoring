from pathlib import Path
import json
import shutil
import tempfile

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from Backend.detect import analyze_image

app = FastAPI(
    title="AI Classroom Monitoring API"
)

# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STUDENTS_FILE = RESULTS_DIR / "students.json"

# ==================================================
# HELPERS
# ==================================================

def load_json(file_path, default):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(
            data,
            f,
            indent=4
        )


def load_score(file_name, default=0):
    try:
        return float(
            np.load(
                RESULTS_DIR / file_name
            )[0]
        )
    except Exception:
        return default

# ==================================================
# ANALYTICS
# ==================================================

@app.get("/dashboard")
def dashboard():

    final_score = load_score(
        "final_score.npy",
        0
    )

    attention_score = load_score(
        "attention_score.npy",
        0
    )

    alertness_score = load_score(
        "alertness_score.npy",
        50
    )

    if final_score >= 80:
        status = "Highly Attentive"

    elif final_score >= 60:
        status = "Moderately Attentive"

    else:
        status = "Needs Attention"

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


@app.get("/analytics")
def analytics():

    analytics_file = (
        RESULTS_DIR /
        "analytics.json"
    )

    analytics = load_json(
        analytics_file,
        {}
    )

    return {
        "attentive_students":
            analytics.get(
                "attentive",
                0
            ),

        "distracted_students":
            analytics.get(
                "distracted",
                0
            ),

        "sleepy_students":
            analytics.get(
                "sleepy",
                0
            ),

        "phone_usage":
            analytics.get(
                "phone",
                0
            ),

        "raw":
            analytics
    }


@app.get("/attention-history")
def attention_history():

    file_path = (
        RESULTS_DIR /
        "attention_history.json"
    )

    return load_json(
        file_path,
        {
            "labels": [],
            "attention": [],
            "engagement": [],
            "pulse": []
        }
    )


@app.get("/detections")
def detections():

    analytics_file = (
        RESULTS_DIR /
        "analytics.json"
    )

    return load_json(
        analytics_file,
        {}
    )

# ==================================================
# STUDENTS
# ==================================================

@app.get("/students")
def students():

    return load_json(
        STUDENTS_FILE,
        {}
    )


@app.post("/register-student")
async def register_student(
    student_id: str = Form(...),
    student_name: str = Form(...),
    file: UploadFile = File(...)
):

    students = load_json(
        STUDENTS_FILE,
        {}
    )

    uploads_dir = (
        BASE_DIR /
        "uploads" /
        "students"
    )

    uploads_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = (
        uploads_dir /
        f"{student_id}_{file.filename}"
    )

    with open(
        image_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    students[student_id] = {

        "name":
            student_name,

        "image":
            str(
                image_path
            ),

        "attention_rate":
            0,

        "status":
            "Registered"
    }

    save_json(
        STUDENTS_FILE,
        students
    )

    return {
        "message":
            "Student registered successfully",

        "student_id":
            student_id,

        "student_name":
            student_name
    }

# ==================================================
# IMAGE ANALYSIS
# ==================================================

@app.post("/analyze-image")
async def analyze_image_api(
    file: UploadFile = File(...)
):
    suffix = Path(
        file.filename
    ).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        content = await file.read()

        temp.write(content)

        temp_path = temp.name

    result = analyze_image(
        temp_path
    )

    return result


# ==================================================
# TEACHER ALERT
# ==================================================

@app.get("/teacher-alert")
def teacher_alert():

    final_score = load_score(
        "final_score.npy",
        0
    )

    if final_score >= 80:

        return {
            "alert_color":
                "GREEN",

            "score":
                final_score,

            "status":
                "Classroom Performing Well"
        }

    elif final_score >= 60:

        return {
            "alert_color":
                "YELLOW",

            "score":
                final_score,

            "status":
                "Moderate Attention"
        }

    else:

        return {
            "alert_color":
                "RED",

            "score":
                final_score,

            "status":
                "Immediate Intervention Recommended"
        }


@app.get("/")
def home():

    return {
        "message":
            "AI Classroom Monitoring API Running"
    }
