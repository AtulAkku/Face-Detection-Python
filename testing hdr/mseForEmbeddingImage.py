import cv2
import numpy as np
import pywt

# Load the original image (before watermarking)
original_image = cv2.imread('your-image~1.jpg', cv2.IMREAD_COLOR)

# Load the embedding image (watermarked image)
embedding_image = cv2.imread('watermarked_image.jpg', cv2.IMREAD_COLOR)

# Ensure both images have the same dimensions
original_image = cv2.resize(original_image, (embedding_image.shape[1], embedding_image.shape[0]))

# Split the images into their RGB channels
original_channels = cv2.split(original_image)
embedding_channels = cv2.split(embedding_image)

wavelets = ['haar', 'db2', 'sym2']  # You can add more wavelets if needed

for wavelet in wavelets:
    print(f'Using {wavelet} wavelet:')
    
    for i in range(3):  # Iterate over each color channel (0: Blue, 1: Green, 2: Red)
        # Apply wavelet transformation to the original and embedding channels
        coeffs_original = pywt.wavedec2(original_channels[i], wavelet, level=1)
        coeffs_embedding = pywt.wavedec2(embedding_channels[i], wavelet, level=1)

        # Calculate the MSE for each channel
        mse = np.mean((coeffs_original[0] - coeffs_embedding[0]) ** 2)
        
        # Print the MSE value for the current channel and wavelet
        channel_name = ['Blue', 'Green', 'Red'][i]
        print(f'MSE ({channel_name}): {mse}')
    
    print()
