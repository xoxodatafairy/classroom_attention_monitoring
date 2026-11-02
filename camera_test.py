import cv2

for i in range(10):

    print(f"Testing camera index {i}")

    cap = cv2.VideoCapture(i)

    if cap.isOpened():

        print(f"✅ Camera {i} detected")

        ret, frame = cap.read()

        print(
            f"Frame read: {ret}"
        )

        cap.release()

    else:

        print(
            f"❌ Camera {i} not available"
        )