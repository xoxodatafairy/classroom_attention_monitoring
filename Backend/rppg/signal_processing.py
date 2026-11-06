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

Notes
-----
- Nothing runs on import any more. Call analyze_signal() with a signal
  (e.g. the "green" list returned by rppg.process_video), or
  analyze_saved_signal() to read the .npy files.
- plot_waveform() returns a matplotlib Figure. Use st.pyplot(fig) in
  Streamlit. plt.show() is only called when run from the terminal.

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

try:
    from .alertness import (
        calculate_alertness_score,
        get_alertness_status
    )
except ImportError:  # running as a plain script
    from alertness import (
        calculate_alertness_score,
        get_alertness_status
    )

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Same folder that rppg.py saves to
RESULTS_DIR = BASE_DIR.parent / "results"


# ==================================================
# MOVING AVERAGE FILTER
# ==================================================

def moving_average(data, window=7):

    smoothed = []

    for i in range(len(data)):

        start = max(0, i - window + 1)
        values = data[start:i + 1]

        smoothed.append(sum(values) / len(values))

    return np.array(smoothed)


# ==================================================
# NORMALIZE
# ==================================================

def normalize(signal):

    signal = np.asarray(signal, dtype=float)

    signal = signal - np.mean(signal)

    std = np.std(signal)

    if std > 0:
        signal = signal / std

    return signal


# ==================================================
# PEAK DETECTION
# ==================================================

def detect_peaks(signal, fps, threshold_factor=0.8, min_gap_seconds=0.4):

    threshold = np.mean(signal) + (threshold_factor * np.std(signal))

    min_distance = max(1, int(fps * min_gap_seconds))

    peaks = []
    last_peak = -999

    for i in range(1, len(signal) - 1):

        is_peak = (
            signal[i] > signal[i - 1]
            and signal[i] > signal[i + 1]
            and signal[i] > threshold
        )

        if is_peak and (i - last_peak) > min_distance:
            peaks.append(i)
            last_peak = i

    return np.array(peaks, dtype=int)


# ==================================================
# ANALYZE SIGNAL
# ==================================================

def analyze_signal(
    signal,
    fps,
    signal_quality=85,
    face_visibility=100,
    save=True,
):
    """
    Normalize, smooth, detect peaks, estimate BPM and alertness.

    Returns a dict, or None if the signal is too short to analyze.
    """

    if signal is None or len(signal) < 10:
        print("⚠️ Signal too short to analyze.")
        return None

    if not fps or fps <= 0:
        fps = 30.0

    print("\n========== ANALYZING SIGNAL ==========")
    print("Samples:", len(signal))
    print("Video FPS:", fps)

    # Normalize + smooth
    signal = normalize(signal)
    signal = moving_average(signal, window=7)

    # Peaks
    peaks = detect_peaks(signal, fps)

    print("Detected Peaks:", len(peaks))

    # BPM
    duration_seconds = len(signal) / fps

    raw_bpm = (len(peaks) / duration_seconds) * 60

    # Clamp to a plausible human range
    estimated_bpm = max(40, min(raw_bpm, 180))

    print(f"Estimated BPM: {estimated_bpm:.1f} (raw {raw_bpm:.1f})")

    # Alertness
    alertness_score = calculate_alertness_score(
        bpm=estimated_bpm,
        signal_quality=signal_quality,
        face_visibility=face_visibility
    )

    status = get_alertness_status(alertness_score)

    print(f"Alertness Score: {alertness_score}")
    print(f"Status: {status}")

    # Save
    if save:

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        alertness_file = RESULTS_DIR / "alertness_score.npy"

        np.save(alertness_file, np.array([alertness_score]))

        print("✅ Alertness Score Saved:", alertness_file)

    return {
        "signal": signal,
        "peaks": peaks,
        "fps": float(fps),
        "bpm": float(estimated_bpm),
        "raw_bpm": float(raw_bpm),
        "alertness_score": alertness_score,
        "status": status,
    }


# ==================================================
# LOAD SAVED SIGNAL
# ==================================================

def analyze_saved_signal(**kwargs):
    """Load green_signal.npy and video_fps.npy from RESULTS_DIR."""

    signal_file = RESULTS_DIR / "green_signal.npy"
    fps_file = RESULTS_DIR / "video_fps.npy"

    if not signal_file.exists() or not fps_file.exists():
        raise FileNotFoundError(
            f"Run rppg.py first. Missing signal files in {RESULTS_DIR}"
        )

    signal = np.load(signal_file)
    fps = float(np.load(fps_file)[0])

    return analyze_signal(signal, fps, **kwargs)


# ==================================================
# PLOT WAVEFORM
# ==================================================

def plot_waveform(result):
    """Return a matplotlib Figure of the pulse waveform."""

    from matplotlib.figure import Figure

    signal = result["signal"]
    peaks = result["peaks"]

    # Figure() (not plt.figure) is safe to use from Streamlit threads
    fig = Figure(figsize=(12, 5))
    ax = fig.subplots()

    ax.plot(
        signal,
        color="green",
        linewidth=1.5,
        label="Pulse Waveform"
    )

    if len(peaks) > 0:
        ax.scatter(
            peaks,
            signal[peaks],
            color="red",
            s=30,
            label="Detected Peaks"
        )

    ax.set_title("rPPG Pulse Waveform")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Normalized Signal")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    return fig


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    result = analyze_saved_signal()

    if result is not None:

        signal = result["signal"]
        peaks = result["peaks"]

        plt.figure(figsize=(12, 5))
        plt.plot(signal, color="green", linewidth=1.5, label="Pulse Waveform")

        if len(peaks) > 0:
            plt.scatter(
                peaks,
                signal[peaks],
                color="red",
                s=30,
                label="Detected Peaks"
            )

        plt.title("rPPG Pulse Waveform")
        plt.xlabel("Frame Number")
        plt.ylabel("Normalized Signal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\n========== FINAL REPORT ==========")
        print(f"Signal Samples: {len(signal)}")
        print(f"Detected Peaks: {len(peaks)}")
        print(f"Estimated BPM: {result['bpm']:.1f}")
        print(f"Alertness Score: {result['alertness_score']}")
        print(f"Status: {result['status']}")

        print("\n✅ Stage Complete")
