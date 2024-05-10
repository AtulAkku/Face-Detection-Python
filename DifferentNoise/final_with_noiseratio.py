import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


facetracker = load_model('facetracker.h5')

cap = cv2.VideoCapture(1)
while cap.isOpened():
  # Check if the frame is empty. If it is, skip the frame.
  ret, frame = cap.read()
  if not ret:
    continue

  # Convert the frame from BGR to RGB color space.
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Resize the frame to 120x120 pixels.
  resized = tf.image.resize(rgb, (120,120))

  # Calculate the noise ratio.
  mean = np.mean(frame)
  std = np.std(frame)
  noise_ratio = std / mean

  # Convert the noise ratio to a percentage.
  noise_ratio_percentage = noise_ratio

  # Display the noise ratio on the screen.
  cv2.putText(frame, 'Noise ratio: {}%'.format(noise_ratio_percentage),
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # Make a prediction on the resized frame.
  yhat = facetracker.predict(np.expand_dims(resized/255,0))

  # Get the sample coordinates from the prediction.
  sample_coords = yhat[1][0]

  # Check if the prediction is confident.
  if yhat[0] > 0.5:

    # Draw a rectangle around the face.
    cv2.rectangle(frame,
                  tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)),
                  (255,0,0), 2)

    # Draw a rectangle around the label.
    cv2.rectangle(frame,
                  tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                              [0,-30])),
                  tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                              [80,0])),
                  (255,0,0), -1)

    # Render the text label.
    cv2.putText(frame, 'face',
                  tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                              [0,-5])),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

  # Display the frame.
  cv2.imshow('FaceTrack', frame)

  # Check if the user pressed the 'q' key to quit.
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the camera and close all windows.
cap.release()
cv2.destroyAllWindows()
