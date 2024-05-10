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

def detect_poisson_noise(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the standard deviation of the pixel values in the grayscale image
    std_dev = np.std(gray_image)

    # Threshold the image based on the standard deviation to identify regions with high Poisson noise
    _, thresholded = cv2.threshold(gray_image, std_dev, 255, cv2.THRESH_BINARY)

    return thresholded

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

cap = cv2.VideoCapture(1)

# Define the size of the smaller frame
small_frame_size = (640, 480)  # You can adjust this size as needed

while cap.isOpened():
    _, frame = cap.read()

    # Resize the frame to the smaller frame size
    frame = cv2.resize(frame, small_frame_size)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker.predict(np.expand_dims(resized/255, 0))
    sample_coords = yhat[1][0]

    # Detect Poisson noise
    poisson_noise_mask = detect_poisson_noise(frame)

    # Detect salt and pepper noise
    salt_pepper_noise_mask = detect_salt_pepper_noise(frame)

    # Detect Gaussian noise
    gaussian_noise_mask = detect_gaussian_noise(frame)

    # Show each noise type in separate windows
    cv2.imshow('Gaussian Noise', gaussian_noise_mask)
    cv2.imshow('Salt & Pepper Noise', salt_pepper_noise_mask)
    cv2.imshow('Poisson Noise', poisson_noise_mask)

    # Show the FaceTrack window with the smaller frame
    cv2.imshow('FaceTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
