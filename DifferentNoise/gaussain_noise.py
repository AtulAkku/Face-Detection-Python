import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def detect_gaussian_noise(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Calculate the absolute difference between the original and blurred images
    diff = cv2.absdiff(gray_image, blurred)

    # Threshold the difference image to identify regions with high Gaussian noise
    _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    return thresholded

facetracker = load_model('facetracker.h5')

cap = cv2.VideoCapture('VID_20220305_220908.mp4')
while cap.isOpened():
    _, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker.predict(np.expand_dims(resized/255, 0))
    sample_coords = yhat[1][0]

    # Detect Gaussian noise
    noise_mask = detect_gaussian_noise(frame)

    # Show noise on screen
    cv2.imshow('Gaussian Noise', noise_mask)

    cv2.imshow('FaceTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
