import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def detect_salt_pepper_noise(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(gray_image, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    noise = gray_image - dilation
    _, thresholded = cv2.threshold(noise, 30, 255, cv2.THRESH_BINARY)
    return thresholded

facetracker = load_model('facetracker.h5')

cap = cv2.VideoCapture(1)
while cap.isOpened():
    _, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker.predict(np.expand_dims(resized/255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        # Controls the main rectangle
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                      (255, 0, 0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),[0, -30])),
                      tuple(np.add(np.multiply(sample_coords[2:], [450, 450]).astype(int),[80, 0])), 
                      (255, 0, 0), -1)

        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                               [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Detect salt and pepper noise
    noise_mask = detect_salt_pepper_noise(frame)

    # Show noise on screen
    cv2.imshow('Salt & Pepper Noise', noise_mask)

    cv2.imshow('FaceTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
