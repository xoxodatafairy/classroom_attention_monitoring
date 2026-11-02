"""
live_detector.py

Real-Time YOLO Detection Engine

Supports:
- Laptop Webcam
- Android Phone Camera
- IP Webcam
- CCTV RTSP Streams

Outputs:
- analytics.json
- attention_score.npy
- final_score.npy
"""

from pathlib import Path
import json

import cv2
import numpy as np

from ultralytics import YOLO

from Backend.scoring.scoring import (
    calculate_attention_score
)

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR /
    "models" /
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
# LOAD MODEL ONCE
# ==================================================

model = YOLO(
    str(MODEL_PATH)
)

# ==================================================
# HELPERS
# ==================================================

def load_alertness_score():

    try:

        return float(
            np.load(
                RESULTS_DIR /
                "alertness_score.npy"
            )[0]
        )

    except Exception:

        return 50.0


def calculate_final_score(
    attention_score,
    alertness_score
):

    final_score = (

        (attention_score * 0.7)

        +

        (alertness_score * 0.3)

    )

    return round(
        final_score,
        2
    )


# ==================================================
# SINGLE FRAME PROCESSING
# ==================================================

def process_frame(frame):

    results = model.predict(

        frame,

        conf=0.30,

        verbose=False
    )

    class_counts = {}

    for result in results:

        if result.boxes is None:

            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            class_name = model.names[
                class_id
            ]

            class_counts[class_name] = (

                class_counts.get(
                    class_name,
                    0
                )

                + 1
            )

    # --------------------------------------
    # Attention Score
    # --------------------------------------

    attention_score = (
        calculate_attention_score(
            class_counts
        )
    )

    # --------------------------------------
    # Alertness Score
    # --------------------------------------

    alertness_score = (
        load_alertness_score()
    )

    # --------------------------------------
    # Fusion
    # --------------------------------------

    final_score = (
        calculate_final_score(
            attention_score,
            alertness_score
        )
    )

    # --------------------------------------
    # Save Analytics
    # --------------------------------------

    with open(
        RESULTS_DIR /
        "analytics.json",
        "w"
    ) as f:

        json.dump(
            class_counts,
            f,
            indent=4
        )

    # --------------------------------------
    # Save Scores
    # --------------------------------------

    np.save(
        RESULTS_DIR /
        "attention_score.npy",
        np.array(
            [attention_score]
        )
    )

    np.save(
        RESULTS_DIR /
        "final_score.npy",
        np.array(
            [final_score]
        )
    )

    return {

        "counts":
        class_counts,

        "attention_score":
        attention_score,

        "alertness_score":
        alertness_score,

        "final_score":
        final_score
    }


# ==================================================
# LIVE CAMERA LOOP
# ==================================================

def run_live_detection(
    source=0
):
    """
    source examples:

    Laptop Webcam:
        0

    USB Camera:
        1

    Android IP Webcam:
        http://192.168.1.10:8080/video

    CCTV RTSP:
        rtsp://user:pass@ip/live
    """

    cap = cv2.VideoCapture(
        source
    )

    if not cap.isOpened():

        raise Exception(
            f"Unable to open camera source: {source}"
        )

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        result = process_frame(
            frame
        )

        display_frame = frame.copy()

        cv2.putText(

            display_frame,

            f"Attention: {result['attention_score']}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2
        )

        cv2.putText(

            display_frame,

            f"Final: {result['final_score']}",

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (255, 0, 0),

            2
        )

        cv2.imshow(
            "AI Classroom Monitoring",
            display_frame
        )

        key = cv2.waitKey(
            1
        ) & 0xFF

        if key == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print(
        "Starting Live Classroom Detection..."
    )

    run_live_detection(
        0
    )
    