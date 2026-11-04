import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ==================================================
# PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Backend.scoring.scoring import calculate_attention_score

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ANALYTICS_FILE = RESULTS_DIR / "analytics.json"
HISTORY_FILE = RESULTS_DIR / "attention_history.json"
ALERTNESS_FILE = RESULTS_DIR / "alertness_score.npy"
PULSE_FILE = RESULTS_DIR / "pulse.npy"

# ==================================================
# MODEL LOADING
# ==================================================

_MODEL = None


def get_model():
    global _MODEL

    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        _MODEL = YOLO(str(MODEL_PATH))

    return _MODEL


# ==================================================
# HELPERS
# ==================================================

def safe_load_npy(file_path, default=0.0):
    try:
        return float(np.load(file_path)[0])
    except Exception:
        return default


def get_alertness_score():
    return safe_load_npy(ALERTNESS_FILE, 50.0)


def get_pulse():
    return safe_load_npy(PULSE_FILE, 0.0)


def get_status(final_score):
    if final_score >= 80:
        return "Highly Attentive"

    if final_score >= 60:
        return "Moderately Attentive"

    return "Needs Attention"


# ==================================================
# HISTORY
# ==================================================

def update_history(
    attention_score,
    final_score,
    pulse
):
    history = {
        "labels": [],
        "attention": [],
        "engagement": [],
        "pulse": []
    }

    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            pass

    history["labels"].append(
        datetime.now().strftime("%H:%M")
    )

    history["attention"].append(
        round(attention_score, 2)
    )

    history["engagement"].append(
        round(final_score, 2)
    )

    history["pulse"].append(
        round(pulse, 2)
    )

    max_items = 100

    for key in history:
        history[key] = history[key][-max_items:]

    with open(HISTORY_FILE, "w") as f:
        json.dump(
            history,
            f,
            indent=4
        )


# ==================================================
# ANALYTICS
# ==================================================

def save_analytics(class_counts):
    with open(
        ANALYTICS_FILE,
        "w"
    ) as f:
        json.dump(
            class_counts,
            f,
            indent=4
        )


# ==================================================
# IMAGE ANALYSIS
# ==================================================

def analyze_image(image_path):
    """
    Main detection pipeline
    """

    model = get_model()

    results = model.predict(
        source=str(image_path),
        conf=0.30,
        save=True,
        verbose=False
    )

    class_counts = {}

    for result in results:

        if not hasattr(result, "boxes"):
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])

            class_name = model.names.get(
                class_id,
                str(class_id)
            )

            class_counts[class_name] = (
                class_counts.get(
                    class_name,
                    0
                ) + 1
            )

    # ==================================================
    # ATTENTION
    # ==================================================

    attention_score = calculate_attention_score(
        class_counts
    )

    attention_score = float(
        max(
            0,
            min(
                100,
                attention_score
            )
        )
    )

    # ==================================================
    # ALERTNESS
    # ==================================================

    alertness_score = get_alertness_score()

    # ==================================================
    # PULSE
    # ==================================================

    pulse = get_pulse()

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

    status = get_status(
        final_score
    )

    # ==================================================
    # SAVE
    # ==================================================

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

    save_analytics(
        class_counts
    )

    update_history(
        attention_score,
        final_score,
        pulse
    )

    total_detected = sum(
        class_counts.values()
    )

    response = {
        "timestamp":
            datetime.now().isoformat(),

        "counts":
            class_counts,

        "total_detected":
            total_detected,

        "attention_score":
            round(
                attention_score,
                2
            ),

        "alertness_score":
            round(
                alertness_score,
                2
            ),

        "pulse":
            round(
                pulse,
                2
            ),

        "final_score":
            round(
                final_score,
                2
            ),

        "status":
            status
    }

    print(
        f"[INFO] "
        f"Attention={attention_score:.2f} | "
        f"Alertness={alertness_score:.2f} | "
        f"Pulse={pulse:.2f} | "
        f"Final={final_score:.2f}"
    )

    return response


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":
    test_image = input(
        "Image path: "
    )

    result = analyze_image(
        test_image
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )
