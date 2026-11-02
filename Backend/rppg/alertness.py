"""
alertness.py

Purpose
-------
Convert rPPG measurements into an
alertness score from 0 to 100.

Author
------
Shalom Arogundade

Project
-------
AI Classroom Attention Monitoring System
"""


# ==================================================
# CALCULATE ALERTNESS SCORE
# ==================================================

def calculate_alertness_score(
    bpm,
    signal_quality=80,
    face_visibility=100
):
    """
    Calculate alertness score.

    Parameters
    ----------
    bpm : float
        Estimated heart rate

    signal_quality : float
        Quality of extracted signal (0-100)

    face_visibility : float
        Percentage of frames where face is visible

    Returns
    -------
    float
        Alertness score between 0 and 100
    """

    score = 0

    # ==================================================
    # HEART RATE CONTRIBUTION
    # ==================================================

    if 60 <= bpm <= 100:

        score += 50

    elif 50 <= bpm < 60:

        score += 35

    elif 100 < bpm <= 120:

        score += 35

    else:

        score += 15

    # ==================================================
    # SIGNAL QUALITY CONTRIBUTION
    # ==================================================

    score += signal_quality * 0.30

    # ==================================================
    # FACE VISIBILITY CONTRIBUTION
    # ==================================================

    score += face_visibility * 0.20

    # ==================================================
    # LIMIT TO 0-100
    # ==================================================

    score = max(
        0,
        min(score, 100)
    )

    return round(score, 2)


# ==================================================
# GET STATUS
# ==================================================

def get_alertness_status(score):
    """
    Convert score into readable status.
    """

    if score >= 80:

        return "Highly Alert ✅"

    elif score >= 60:

        return "Moderately Alert ⚠️"

    else:

        return "Low Alertness ❌"


# ==================================================
# TEST MODULE
# ==================================================

if __name__ == "__main__":

    bpm = 75

    score = calculate_alertness_score(
        bpm=bpm,
        signal_quality=85,
        face_visibility=100
    )

    status = get_alertness_status(score)

    print("\n========== ALERTNESS ==========")

    print("BPM:", bpm)
    print("Score:", score)
    print("Status:", status)