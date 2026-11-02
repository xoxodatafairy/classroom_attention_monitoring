from pathlib import Path
import json
import random

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    exist_ok=True
)

students = {

    "ID12900": {

        "name": "Student 1",

        "attention_rate": random.randint(
            60,
            95
        ),

        "status": "Attentive"
    },

    "ID12901": {

        "name": "Student 2",

        "attention_rate": random.randint(
            40,
            80
        ),

        "status": "Distracted"
    },

    "ID12902": {

        "name": "Student 3",

        "attention_rate": random.randint(
            50,
            90
        ),

        "status": "Moderate"
    }
}

with open(
    RESULTS_DIR /
    "student_metrics.json",
    "w"
) as f:

    json.dump(
        students,
        f,
        indent=4
    )

print(
    "student_metrics.json created"
)