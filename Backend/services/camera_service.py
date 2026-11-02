import cv2

def open_camera():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        raise Exception(
            "Unable to access webcam"
        )

    return camera