"""
rppg.py

Purpose
-------
Remote Photoplethysmography (rPPG) pipeline

Steps
-----
1. Load classroom video
2. Detect faces
3. Select largest face per frame
4. Extract forehead ROI (clamped to the frame)
5. Extract RGB signal
6. Smooth signal
7. Save signal for BPM estimation

Notes
-----
- No OpenCV GUI calls (imshow / waitKey) unless show_preview=True.
  GUI calls from Streamlit's worker thread crash on macOS with
  "Unknown C++ exception from OpenCV code".
- process_video() returns its results, so the dashboard does not
  have to re-read files from disk.

Author
------
Shalom Arogundade

Project
-------
AI Classroom Attention Monitoring System
"""

# ==================================================
# IMPORT LIBRARIES
# ==================================================

import traceback
from pathlib import Path

import cv2
import numpy as np

try:
    from .face_detector import detect_faces
except ImportError:  # running as a plain script
    from face_detector import detect_faces

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Backend/results (same folder your dashboard already uses)
RESULTS_DIR = BASE_DIR / "results"

VIDEO_PATH = BASE_DIR / "videos" / "classroom.mp4"


# ==================================================
# MOVING AVERAGE FILTER
# ==================================================

def moving_average(signal, window=5):
    """Smooth a signal using a trailing moving average."""

    smoothed = []

    for i in range(len(signal)):

        start = max(0, i - window + 1)
        values = signal[start:i + 1]

        smoothed.append(sum(values) / len(values))

    return smoothed


# ==================================================
# FOREHEAD ROI (CLAMPED)
# ==================================================

def get_forehead_box(face, frame_shape):
    """
    Return (x1, y1, x2, y2) of the forehead ROI, clamped to the
    frame, or None if the ROI is empty or too small.
    """

    x, y, w, h = [int(v) for v in face]

    frame_h, frame_w = frame_shape[:2]

    x1 = x + int(w * 0.25)
    y1 = y + int(h * 0.10)
    x2 = x1 + int(w * 0.50)
    y2 = y1 + int(h * 0.20)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return None

    return x1, y1, x2, y2


# ==================================================
# PROCESS VIDEO
# ==================================================

def process_video(
    video_path,
    show_preview=False,
    preview_every=30,
    max_preview_frames=6,
    save=True,
):
    """
    Run the rPPG pipeline on a video.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    show_preview : bool
        Open an OpenCV window. Only use when running this file
        directly from the terminal, never inside Streamlit.
    preview_every : int
        Keep an annotated preview frame every N frames.
    max_preview_frames : int
        Maximum number of preview frames to keep.
    save : bool
        Save green_signal.npy and video_fps.npy to RESULTS_DIR.

    Returns
    -------
    dict with keys: fps, green, red, blue, frames, faces, preview
    """

    video_path = str(video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    print("\nVideo FPS:", fps)

    frame_count = 0
    total_faces_detected = 0
    skipped_frames = 0

    green_signal = []
    red_signal = []
    blue_signal = []

    preview_frames = []

    print("\n========== PROCESSING VIDEO ==========")

    try:

        while True:

            success, frame = cap.read()

            if not success or frame is None:
                break

            frame_count += 1

            try:

                # ------------------------------------------
                # Detect faces
                # ------------------------------------------

                faces = detect_faces(frame)

                total_faces_detected += len(faces)

                if len(faces) > 0:

                    # --------------------------------------
                    # Largest face only
                    # --------------------------------------

                    largest_face = max(
                        faces,
                        key=lambda face: face[2] * face[3]
                    )

                    x, y, w, h = [int(v) for v in largest_face]

                    # --------------------------------------
                    # Forehead ROI -> RGB signal
                    # --------------------------------------

                    box = get_forehead_box(
                        largest_face,
                        frame.shape
                    )

                    if box is not None:

                        x1, y1, x2, y2 = box

                        forehead = np.ascontiguousarray(
                            frame[y1:y2, x1:x2],
                            dtype=np.uint8
                        )

                        if forehead.size > 0:

                            # OpenCV frames are BGR
                            blue_signal.append(
                                float(np.mean(forehead[:, :, 0]))
                            )
                            green_signal.append(
                                float(np.mean(forehead[:, :, 1]))
                            )
                            red_signal.append(
                                float(np.mean(forehead[:, :, 2]))
                            )

                        # Draw forehead box
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            "Forehead ROI",
                            (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1
                        )

                    # Draw face box
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

            except cv2.error:

                # One bad frame should not kill the whole run
                skipped_frames += 1

                if skipped_frames <= 3:
                    traceback.print_exc()

                continue

            # ----------------------------------------------
            # Preview frames for the dashboard (RGB)
            # ----------------------------------------------

            if (
                frame_count % preview_every == 0
                and len(preview_frames) < max_preview_frames
            ):
                preview_frames.append(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

            # ----------------------------------------------
            # Optional OpenCV window (terminal use only)
            # ----------------------------------------------

            if show_preview:

                cv2.imshow("rPPG Face Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:

        cap.release()

        if show_preview:
            cv2.destroyAllWindows()

    # ==================================================
    # SMOOTH SIGNAL
    # ==================================================

    smoothed_green = moving_average(green_signal, window=5)

    # ==================================================
    # SAVE SIGNALS
    # ==================================================

    if save and len(smoothed_green) > 0:

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        np.save(
            RESULTS_DIR / "green_signal.npy",
            np.array(smoothed_green)
        )

        np.save(
            RESULTS_DIR / "video_fps.npy",
            np.array([fps])
        )

        print("\n✅ green_signal.npy saved")
        print("✅ video_fps.npy saved")
        print("Saved to:", RESULTS_DIR)

    elif save:

        print("\n⚠️ No face/forehead samples collected. Nothing saved.")

    # ==================================================
    # REPORT
    # ==================================================

    print("\n========== RESULTS ==========")
    print("Frames Processed:", frame_count)
    print("Faces Detected:", total_faces_detected)
    print("Signal Samples:", len(smoothed_green))
    print("Skipped Frames (OpenCV errors):", skipped_frames)

    return {
        "fps": float(fps),
        "green": smoothed_green,
        "red": red_signal,
        "blue": blue_signal,
        "frames": frame_count,
        "faces": total_faces_detected,
        "preview": preview_frames,
    }


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":

    print("\n========== VIDEO CHECK ==========")
    print("Video:", VIDEO_PATH)
    print("Exists:", VIDEO_PATH.exists())

    process_video(VIDEO_PATH, show_preview=True)
