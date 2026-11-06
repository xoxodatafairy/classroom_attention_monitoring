"""
face_detector.py

Purpose
-------
Detect faces inside video frames and extract
forehead regions for rPPG processing.

Author
------
Shalom Arogundade
"""

import cv2

# ==================================================
# LOAD FACE DETECTOR
# ==================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==================================================
# DETECT FACES
# ==================================================

def detect_faces(frame):

    # Convert image to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    return faces

# ==================================================
# EXTRACT FOREHEAD ROI
# ==================================================
    """
    Extract the forehead region from a detected face.

    Parameters
    ----------
    frame : ndarray
        Video frame

    face : tuple
        (x, y, w, h)

    Returns
    -------
    ndarray
        Forehead region of interest
    """

def extract_forehead(frame, face):

    x, y, w, h = face

    forehead_x = x + int(w * 0.25)
    forehead_y = y + int(h * 0.10)

    forehead_w = int(w * 0.50)
    forehead_h = int(h * 0.20)

    return frame[
        forehead_y:forehead_y + forehead_h,
        forehead_x:forehead_x + forehead_w
    ]

    return forehead