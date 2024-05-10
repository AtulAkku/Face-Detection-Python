import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pywt

# Load the face tracking model
facetracker = load_model('facetracker.h5')

# Define the wavelet to use (e.g., Haar wavelet)
wavelet = 'haar'

# Initialize the camera capture
cap = cv2.VideoCapture(1)

while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500, :]

    # Convert the frame to a 3-channel image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform wavelet transformation
    coefficients = pywt.wavedec2(frame_rgb, wavelet)

    # Reconstruct the 3-channel image from wavelet coefficients
    reconstructed_image = cv2.merge((coefficients[0], coefficients[0], coefficients[0]))

    # Normalize the reconstructed image
    reconstructed_image = reconstructed_image / 255.0

    yhat = facetracker.predict(np.expand_dims(reconstructed_image, 0))

    if yhat[0] > 0.5:
        # Controls the main rectangle
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                      (255, 0, 0), 2)

        # Controls the label rectangle
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                    [0, -30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                    [80, 0]),
                            (255, 0, 0), -1))

        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                              [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('FaceTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
