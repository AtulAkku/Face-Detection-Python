import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def detect_salt_pepper_noise(image):
    # Your salt and pepper noise detection code here
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


# Function to calculate noise level in an image
def calculate_noise_level(noise_mask):
    white_pixel_count = np.sum(noise_mask == 255)
    total_pixels = noise_mask.size
    noise_level = (white_pixel_count / total_pixels) * 100
    return noise_level

facetracker = load_model('facetracker.h5')

cap = cv2.VideoCapture('VID_20220305_220908.mp4')

small_frame_size = (480, 640)

while cap.isOpened():
    _, frame = cap.read()
    frame_resized = cv2.resize(frame, small_frame_size)

    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker.predict(np.expand_dims(resized/255, 0))
    sample_coords = yhat[1][0]

    # Detect Poisson noise
    poisson_noise_mask = detect_poisson_noise(frame_resized)
    poisson_noise_level = calculate_noise_level(poisson_noise_mask)
    text_color = (0, 0, 0)  # Black text for white noise mask
    cv2.putText(poisson_noise_mask, f'Poisson Noise: {poisson_noise_level:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Detect salt and pepper noise
    salt_pepper_noise_mask = detect_salt_pepper_noise(frame_resized)
    salt_pepper_noise_level = calculate_noise_level(salt_pepper_noise_mask)
    text_color = (255, 255, 255)  # White text for dark noise mask
    cv2.putText(salt_pepper_noise_mask, f'Salt & Pepper Noise: {salt_pepper_noise_level:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Detect Gaussian noise
    gaussian_noise_mask = detect_gaussian_noise(frame_resized)
    gaussian_noise_level = calculate_noise_level(gaussian_noise_mask)
    text_color = (255, 255, 255)  # White text for dark noise mask
    cv2.putText(gaussian_noise_mask, f'Gaussian Noise: {gaussian_noise_level:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Show each noise type in separate windows
    cv2.imshow('Gaussian Noise', gaussian_noise_mask)
    cv2.imshow('Salt & Pepper Noise', salt_pepper_noise_mask)
    cv2.imshow('Poisson Noise', poisson_noise_mask)

    # Show the FaceTrack window
    cv2.imshow('FaceTrack', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
