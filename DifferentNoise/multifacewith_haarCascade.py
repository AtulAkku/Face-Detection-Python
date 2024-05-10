import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

facetracker = load_model('facetracker.h5')

# Load a pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)  # Change the camera index to 0 for your default camera
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Crop the detected face
        face_roi = frame[y:y + h, x:x + w]
        resized = tf.image.resize(face_roi, (120, 120))
        
        yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
        
        if yhat[0] > 0.5:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Draw a label rectangle
            cv2.rectangle(frame, (x, y - 30), (x + 80, y), (255, 0, 0), -1)
            # Display "face" label
            cv2.putText(frame, 'face', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('FaceTracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
