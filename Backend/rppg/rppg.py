"""
rgnal
6. Smooth signalppg.py

Purpose
-------
Remote Photoplethysmography (rPPG) Pipeline

Features
--------
1. Load classroom video
2. Detect faces
3. Select largest face per frame
4. Extract forehead ROI
5. Extract RGB si
7. Save signal for BPM estimation

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

from pathlib import Path
import cv2
import numpy as np

from face_detector import (
    detect_faces,
    extract_forehead
)

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    BASE_DIR /
    "videos" /
    "classroom.mp4"
)

print("\n========== VIDEO CHECK ==========")
print("Video:", VIDEO_PATH)
print("Exists:", VIDEO_PATH.exists())


# ==================================================
# MOVING AVERAGE FILTER
# ==================================================

def moving_average(signal, window=5):
    """
    Smooth signal using moving average.
    """

    smoothed = []

    for i in range(len(signal)):

        start = max(0, i - window + 1)

        values = signal[start:i + 1]

        smoothed.append(
            sum(values) / len(values)
        )

    return smoothed


# ==================================================
# PROCESS VIDEO
# ==================================================

def process_video(video_path):

    # ----------------------------------------------
    # Open video
    # ----------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        raise ValueError(
            f"Cannot open video: {video_path}"
        )

    # ----------------------------------------------
    # Get video FPS
    # ----------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print("\nVideo FPS:", fps)

    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    frame_count = 0
    total_faces_detected = 0

    # ----------------------------------------------
    # Signal storage
    # ----------------------------------------------

    green_signal = []
    red_signal = []
    blue_signal = []

    print("\n========== PROCESSING VIDEO ==========")

    # ----------------------------------------------
    # Process frames
    # ----------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # ==================================================
        # STEP 1: DETECT FACES
        # ==================================================

        faces = detect_faces(frame)

        total_faces_detected += len(faces)

        # ==================================================
        # STEP 2: USE LARGEST FACE ONLY
        # ==================================================

        if len(faces) > 0:

            largest_face = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            x, y, w, h = largest_face

            # --------------------------------------------------
            # Extract forehead ROI
            # --------------------------------------------------

            forehead = extract_forehead(
                frame,
                largest_face
            )

            # --------------------------------------------------
            # Extract RGB signal
            # --------------------------------------------------

            if forehead.size > 0:

                mean_blue = np.mean(
                    forehead[:, :, 0]
                )

                mean_green = np.mean(
                    forehead[:, :, 1]
                )

                mean_red = np.mean(
                    forehead[:, :, 2]
                )

                blue_signal.append(
                    float(mean_blue)
                )

                green_signal.append(
                    float(mean_green)
                )

                red_signal.append(
                    float(mean_red)
                )

            # --------------------------------------------------
            # Draw face rectangle
            # --------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # --------------------------------------------------
            # Forehead coordinates
            # --------------------------------------------------

            forehead_x = x + int(w * 0.25)
            forehead_y = y + int(h * 0.10)

            forehead_w = int(w * 0.50)
            forehead_h = int(h * 0.20)

            # --------------------------------------------------
            # Draw forehead rectangle
            # --------------------------------------------------

            cv2.rectangle(
                frame,
                (
                    forehead_x,
                    forehead_y
                ),
                (
                    forehead_x + forehead_w,
                    forehead_y + forehead_h
                ),
                (0, 0, 255),
                2
            )

            # --------------------------------------------------
            # Add label
            # --------------------------------------------------

            cv2.putText(
                frame,
                "Forehead ROI",
                (
                    forehead_x,
                    forehead_y - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        # ==================================================
        # STEP 3: DISPLAY VIDEO
        # ==================================================

        cv2.imshow(
            "rPPG Face Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ==================================================
    # CLEANUP
    # ==================================================

    cap.release()
    cv2.destroyAllWindows()

    # ==================================================
    # STEP 4: SMOOTH SIGNAL
    # ==================================================

    smoothed_green = moving_average(
        green_signal,
        window=5
    )

    # ==================================================
    # STEP 5: SAVE SIGNALS
    # ==================================================

    np.save(
        "green_signal.npy",
        np.array(smoothed_green)
    )

    np.save(
        "video_fps.npy",
        np.array([fps])
    )

    print("\n✅ green_signal.npy saved")
    print("✅ video_fps.npy saved")

    # ==================================================
    # REPORT RGB SIGNAL
    # ==================================================

    print("\n========== RGB SIGNAL ==========")

    print(
        "Samples Collected:",
        len(smoothed_green)
    )

    if len(smoothed_green) > 0:

        print(
            "\nFirst Green Samples:"
        )

        print(
            smoothed_green[:10]
        )

    # ==================================================
    # FINAL REPORT
    # ==================================================

    print("\n========== RESULTS ==========")

    print(
        "Frames Processed:",
        frame_count
    )

    print(
        "Faces Detected:",
        total_faces_detected
    )

    print(
        "Signal Samples:",
        len(smoothed_green)
    )

    print("\n✅ Video Loading")
    print("✅ Face Detection")
    print("✅ Largest Face Selection")
    print("✅ Forehead ROI Extraction")
    print("✅ RGB Extraction")
    print("✅ Signal Smoothing")
    print("✅ Signal Saved")


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":

    process_video(
        str(VIDEO_PATH)
    )