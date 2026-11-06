import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from Backend.scoring.scoring import (
    calculate_attention_score
)

from Backend.fusion.fusion import (
    calculate_final_score,
    get_classroom_status
)
# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "best.pt"
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("MODEL_PATH =", MODEL_PATH)
print("EXISTS =", MODEL_PATH.exists())
# ==================================================
# LOAD MODEL
# ==================================================

model = YOLO(str(MODEL_PATH))
# ==================================================
# ANALYZE IMAGE
# ==================================================
def analyze_image(image_path):

    results = model.predict(
        source=str(image_path),
        conf=0.30,
        save=True
    )

    class_counts = {}

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            class_counts[class_name] = (
                class_counts.get(class_name, 0) + 1
            )

    attention_score = calculate_attention_score(
        class_counts
    )

    attention_score = max(
        0,
        min(100, attention_score)
    )

    np.save(
        RESULTS_DIR / "attention_score.npy",
        np.array([attention_score])
    )

    try:
        alertness_score = float(
            np.load(
                RESULTS_DIR / "alertness_score.npy"
            )[0]
        )
    except Exception:
        alertness_score = 50.0

    final_score = round(
        (attention_score * 0.7)
        + (alertness_score * 0.3),
        2
    )

    status = get_classroom_status(
        final_score
    )

    return {
        "counts": class_counts,
        "attention_score": round(attention_score, 2),
        "alertness_score": round(alertness_score, 2),
        "final_score": final_score,
        "status": status
    }
def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    aggregated_counts = {}
    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Analyze every 30th frame
        if frame_count % 90 != 0:
            continue

        results = model.predict(
            source=frame,
            conf=0.90,
            verbose=False
        )

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                aggregated_counts[class_name] = (
                    aggregated_counts.get(
                        class_name,
                        0
                    ) + 1
                )

    cap.release()

    attention_score = calculate_attention_score(
        aggregated_counts
    )

    try:
        from Backend.rppg.rppg import process_video

        process_video(video_path)

        print("✅ rPPG Processing Complete")

    except Exception as e:

        print(
            f"❌ rPPG Error: {e}"
        )

    try:

        alertness_score = float(
            np.load(
                RESULTS_DIR /
                "alertness_score.npy"
            )[0]
        )

    except Exception:

        alertness_score = 50.0

    final_score = round(
        (attention_score * 0.7)
        +
        (alertness_score * 0.3),
        2
    )

    status = get_classroom_status(
        final_score
    )
    # ==========================================
# rPPG PROCESSING
# ==========================================

    try:

        process_video(video_path)

        print("✅ rPPG Processing Complete")

    except Exception as e:

        print(
            f"❌ rPPG Error: {e}"
        )

        attention_score = calculate_attention_score(
            aggregated_counts
        )

    try:
        alertness_score = float(
            np.load(
                RESULTS_DIR /
                "alertness_score.npy"
            )[0]
        )
    except:
        alertness_score = 50.0

    final_score = round(
        (attention_score * 0.7)
        +
        (alertness_score * 0.3),
        2
    )

    status = get_classroom_status(
        final_score
    )

    return {
        "counts": aggregated_counts,
        "attention_score": round(attention_score, 2),
        "alertness_score": round(alertness_score, 2),
        "final_score": round(final_score, 2),
        "status": status
    }
