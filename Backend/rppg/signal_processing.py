"""
signal_processing.py

Purpose
-------
1. Load saved rPPG signal
2. Normalize signal
3. Smooth signal
4. Detect pulse peaks
5. Estimate BPM
6. Calculate alertness score
7. Save alertness score
8. Plot pulse waveform

Author
------
Shalom Arogundade

Project
-------
AI Classroom Attention Monitoring System
"""

# ==================================================
# IMPORTS
# ==================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from alertness import (
    calculate_alertness_score,
    get_alertness_status
)

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# ==================================================
# LOAD SIGNAL FILES
# ==================================================

print("\n========== LOADING SIGNAL ==========")

signal = np.load("green_signal.npy")

fps = float(
    np.load("video_fps.npy")[0]
)

print("Samples:", len(signal))
print("Video FPS:", fps)

# ==================================================
# NORMALIZE SIGNAL
# ==================================================

signal = signal - np.mean(signal)

std = np.std(signal)

if std > 0:

    signal = signal / std

print("✅ Signal Normalized")

# ==================================================
# MOVING AVERAGE FILTER
# ==================================================

def moving_average(data, window=7):

    smoothed = []

    for i in range(len(data)):

        start = max(
            0,
            i - window + 1
        )

        values = data[start:i + 1]

        smoothed.append(
            sum(values) / len(values)
        )

    return np.array(smoothed)

# ==================================================
# SMOOTH SIGNAL
# ==================================================

signal = moving_average(
    signal,
    window=7
)

print("✅ Signal Smoothed")

# ==================================================
# PEAK DETECTION
# ==================================================

peaks = []

threshold = (
    np.mean(signal)
    + (0.8 * np.std(signal))
)

last_peak = -999

min_distance = int(
    fps * 0.4
)

for i in range(1, len(signal) - 1):

    is_peak = (

        signal[i] > signal[i - 1]

        and

        signal[i] > signal[i + 1]

        and

        signal[i] > threshold
    )

    if is_peak and (
        i - last_peak
    ) > min_distance:

        peaks.append(i)

        last_peak = i

peaks = np.array(peaks)

print("\n========== PEAK DETECTION ==========")

print("Detected Peaks:", len(peaks))

# ==================================================
# BPM ESTIMATION
# ==================================================

duration_seconds = len(signal) / fps

estimated_bpm = (
    len(peaks) / duration_seconds
) * 60

estimated_bpm = max(
    40,
    min(
        estimated_bpm,
        180
    )
)

print("\n========== HEART RATE ==========")

print(
    f"Estimated BPM: {estimated_bpm:.1f}"
)

# ==================================================
# CALCULATE ALERTNESS SCORE
# ==================================================

alertness_score = (
    calculate_alertness_score(
        bpm=estimated_bpm,
        signal_quality=85,
        face_visibility=100
    )
)

status = (
    get_alertness_status(
        alertness_score
    )
)

print("\n========== ALERTNESS ==========")

print(
    f"Alertness Score: {alertness_score}"
)

print(
    f"Status: {status}"
)

# ==================================================
# SAVE ALERTNESS SCORE
# ==================================================

alertness_file = (
    RESULTS_DIR /
    "alertness_score.npy"
)

np.save(
    alertness_file,
    np.array([alertness_score])
)

print("\n✅ Alertness Score Saved")

print(alertness_file)

# ==================================================
# PLOT WAVEFORM
# ==================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    signal,
    color="green",
    linewidth=1.5,
    label="Pulse Waveform"
)

if len(peaks) > 0:

    plt.scatter(
        peaks,
        signal[peaks],
        color="red",
        s=30,
        label="Detected Peaks"
    )

plt.title(
    "rPPG Pulse Waveform"
)

plt.xlabel(
    "Frame Number"
)

plt.ylabel(
    "Normalized Signal"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()

# ==================================================
# FINAL REPORT
# ==================================================

print("\n========== FINAL REPORT ==========")

print(
    f"Signal Samples: {len(signal)}"
)

print(
    f"Detected Peaks: {len(peaks)}"
)

print(
    f"Estimated BPM: {estimated_bpm:.1f}"
)

print(
    f"Alertness Score: {alertness_score}"
)

print(
    f"Status: {status}"
)

print("\n✅ Stage Complete")