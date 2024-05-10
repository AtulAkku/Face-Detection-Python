import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def detect_poisson_noise(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the standard deviation of the pixel values in the grayscale image
    std_dev = np.std(gray_image)

    # Threshold the image based on the standard deviation to identify regions with high Poisson noise
    _, thresholded = cv2.threshold(gray_image, std_dev, 255, cv2.THRESH_BINARY)

    return thresholded

facetracker = load_model('facetracker.h5')

cap = cv2.VideoCapture('VID_20220305_220908.mp4')
while cap.isOpened():
    _, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker.predict(np.expand_dims(resized/255, 0))
    sample_coords = yhat[1][0]

    # Detect Poisson noise
    noise_mask = detect_poisson_noise(frame)

    # Show noise on screen
    cv2.imshow('Poisson Noise', noise_mask)

    cv2.imshow('FaceTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
