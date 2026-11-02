"""
scoring.py

Purpose:
Convert detected behaviors into an
overall classroom attention score.

Score Range:
0 - 100
"""


def calculate_attention_score(class_counts):

    # Start score at zero
    score = 0

    # --------------------------------------
    # Positive Behaviors
    # --------------------------------------

    score += class_counts.get(
        "Attentive",
        0
    ) * 10

    score += class_counts.get(
        "hand_rising",
        0
    ) * 8

    # --------------------------------------
    # Negative Behaviors
    # --------------------------------------

    score -= class_counts.get(
        "Distracted",
        0
    ) * 6

    score -= class_counts.get(
        "Sleepy",
        0
    ) * 10

    score -= class_counts.get(
        "phone_use",
        0
    ) * 12

    score -= class_counts.get(
        "daydreaming",
        0
    ) * 8

    score -= class_counts.get(
        "bullying",
        0
    ) * 20

    # --------------------------------------
    # Keep score between 0 and 100
    # --------------------------------------

    score = max(
        0,
        min(score, 100)
    )

    return score


# Test Block

if __name__ == "__main__":

    sample = {

        "Attentive": 8,
        "Distracted": 2,
        "Sleepy": 1,
        "hand_rising": 1

    }

    score = calculate_attention_score(
        sample
    )

    print(
        "Attention Score:",
        score
    )