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
    "Models" /
    "best.pt"
)

RESULTS_DIR = (
    BASE_DIR /
    "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)

# ==================================================
# LOAD YOLO MODEL ONCE
# ==================================================

model = YOLO(
    str(MODEL_PATH)
)

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

            class_id = int(
                box.cls[0]
            )

            class_name = model.names[
                class_id
            ]

            class_counts[
                class_name
            ] = (
                class_counts.get(
                    class_name,
                    0
                ) + 1
            )
def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    aggregated_counts = {}

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 30 != 0:
            continue

        results = model.predict(
            source=frame,
            conf=0.30,
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

    # ==================================================
    # ATTENTION SCORE
    # ==================================================

    attention_score = (
        calculate_attention_score(
            class_counts
        )
    )
    
    
    attention_score = max(
    0,
    min(
        100,
        attention_score
    )
)

    np.save(
        RESULTS_DIR /
        "attention_score.npy",
        np.array(
            [attention_score]
        )
    )

    # ==================================================
    # SAVE ANALYTICS
    # ==================================================

    analytics_file = (
        RESULTS_DIR /
        "analytics.json"
    )

    with open(
        analytics_file,
        "w"
    ) as f:

        json.dump(
            class_counts,
            f,
            indent=4
        )

    # ==================================================
    # LOAD ALERTNESS SCORE
    # ==================================================

    alertness_file = (
        RESULTS_DIR /
        "alertness_score.npy"
    )

    try:

        alertness_score = float(
            np.load(
                alertness_file
            )[0]
        )

    except Exception:

        alertness_score = 50.0

    # ==================================================
    # FINAL SCORE
    # ==================================================

    final_score = round(
    (
        attention_score * 0.7
    )
    +
    (
        alertness_score * 0.3
    ),
    2
)

    final_score = max(
    0,
    min(
        100,
        final_score
    )
)

    np.save(

        RESULTS_DIR /
        "final_score.npy",

        np.array(
            [final_score]
        )
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

    return {
        "counts": aggregated_counts,
        "attention_score": round(
            attention_score,
            2
        ),
        "alertness_score": round(
            alertness_score,
            2
        ),
        "final_score": final_score,
        "status": status
    }

    # ==================================================
    # ATTENTION HISTORY
    # ==================================================

    history_file = (
        RESULTS_DIR /
        "attention_history.json"
    )

    try:

        with open(
            history_file,
            "r"
        ) as f:

            history = json.load(f)

    except Exception:

        history = {

            "labels": [],

            "attention": [],

            "engagement": [],

            "pulse": []
        }

    history["labels"].append(
        datetime.now().strftime(
            "%H:%M"
        )
    )

    history["attention"].append(
        round(
            float(
                attention_score
            ),
            2
        )
    )

    history["engagement"].append(
        round(
            float(
                final_score
            ),
            2
        )
    )

    # Placeholder until rPPG is added
    history["pulse"].append(
        0
    )

    # Keep only latest 100 records

    for key in history:

        history[key] = history[
            key
        ][-100:]

    with open(
        history_file,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )
        

    # ==================================================
    # RETURN RESULTS
    # ==================================================

    status = get_classroom_status(final_score)

    return {
    "counts": class_counts,
    "attention_score": round(attention_score, 2),
    "alertness_score": round(alertness_score, 2),
    "final_score": round(final_score, 2),
    "status": status
}