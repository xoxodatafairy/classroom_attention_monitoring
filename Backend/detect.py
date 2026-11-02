import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Backend.scoring.scoring import (
    calculate_attention_score
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
# LOAD MODEL
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
            float(attention_score)
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
    # ANALYTICS
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
    # ALERTNESS SCORE
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

    if final_score >= 80:

        status = "Highly Attentive"

    elif final_score >= 60:

        status = "Moderately Attentive"

    else:

        status = "Needs Attention"

    np.save(
        RESULTS_DIR /
        "final_score.npy",
        np.array(
            [final_score]
        )
    )

    # ==================================================
    # HISTORY
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
            attention_score,
            2
        )
    )

    history["engagement"].append(
        round(
            final_score,
            2
        )
    )

    history["pulse"].append(
        0
    )

    # Keep latest 100 records

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

    print(
        f"Attention={attention_score}, "
        f"Alertness={alertness_score}, "
        f"Final={final_score}"
    )

    return {

        "counts":
        class_counts,

        "attention_score":
        attention_score,

        "alertness_score":
        alertness_score,

        "final_score":
        final_score,

        "status":
        status
    }