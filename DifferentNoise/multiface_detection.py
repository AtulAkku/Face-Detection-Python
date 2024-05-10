import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# Load the MTCNN model
multiface_detector = MTCNN()

cap = cv2.VideoCapture(1)

while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500, :]

    # Use the multi-face detection model to get face coordinates
    face_coords = multiface_detector.detect_faces(frame)

    for face in face_coords:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y - 30), (x + 80, y), (255, 0, 0), -1)
        cv2.putText(frame, 'face', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Multi-Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
