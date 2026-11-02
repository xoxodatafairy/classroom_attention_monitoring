"""
fusion.py

Purpose
-------
Combine outputs from multiple AI modules
into a single classroom attention score.

Current Inputs
--------------
1. YOLO Attention Score

Future Inputs
-------------
2. rPPG Alertness Score
3. Face Engagement Score
4. Participation Score

Author
------
Shalom Arogundade

Project
-------
AI Classroom Attention Monitoring System
"""
# ==================================================
# FUNCTION: CALCULATE FINAL SCORE
# ==================================================

def calculate_final_score(
    yolo_score,
    rppg_score,
    yolo_weight=0.7,
    rppg_weight=0.3
):
    """
    Combines multiple scores using a weighted average.

    Parameters
    ----------
    yolo_score : float
        Score generated from classroom behaviours.

    rppg_score : float
        Score generated from alertness estimation.

    yolo_weight : float
        Importance of YOLO score.

    rppg_weight : float
        Importance of rPPG score.

    Returns
    -------
    float
        Final classroom score between 0 and 100.
    """

    final_score = (
        (yolo_score * yolo_weight)
        +
        (rppg_score * rppg_weight)
    )

    return round(final_score, 2)


# ==================================================
# FUNCTION: INTERPRET SCORE
# ==================================================

def get_classroom_status(score):
    """
    Converts numeric score into human-readable status.
    """

    if score >= 80:
        return "Highly Attentive ✅"

    elif score >= 60:
        return "Moderately Attentive ⚠️"

    else:
        return "Low Attention ❌"


# ==================================================
# TEST SECTION
# ==================================================

if __name__ == "__main__":

    print("\n========== FUSION TEST ==========\n")

    yolo_score = 66
    rppg_score = 80

    final_score = calculate_final_score(
        yolo_score,
        rppg_score
    )

    status = get_classroom_status(
        final_score
    )

    print("YOLO Score:", yolo_score)
    print("rPPG Score:", rppg_score)

    print("\nFinal Score:", final_score)
    print("Status:", status)