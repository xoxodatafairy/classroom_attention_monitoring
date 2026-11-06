import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent

HISTORY_FILE = (
    BASE_DIR /
    "results" /
    "attention_history.json"
)


def append_history(
    attention_score,
    final_score,
    pulse_score=0
):

    try:

        with open(
            HISTORY_FILE,
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
        datetime.now().strftime("%H:%M")
    )

    history["attention"].append(
        round(attention_score, 2)
    )

    history["engagement"].append(
        round(final_score, 2)
    )

    history["pulse"].append(
        round(pulse_score, 2)
    )

    # keep latest 100 records

    for key in history:

        history[key] = history[key][-100:]

    with open(
        HISTORY_FILE,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )