# test_iphone.py

import cv2

url = "http://172.24.109.143:8081"

cap = cv2.VideoCapture(url)

print("Connected:", cap.isOpened())

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("iPhone Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()