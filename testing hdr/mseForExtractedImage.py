import cv2
import numpy as np
import pywt

# Load the original image (before watermarking)
original_image = cv2.imread('watermarked_image.jpg', cv2.IMREAD_COLOR)

# Load the extracted image (after watermark extraction)
extracted_image = cv2.imread('extracted_watermarked_image.jpg', cv2.IMREAD_COLOR)

# Ensure both images have the same dimensions
original_image = cv2.resize(original_image, (extracted_image.shape[1], extracted_image.shape[0]))

# Split the images into their RGB channels
original_channels = cv2.split(original_image)
extracted_channels = cv2.split(extracted_image)

wavelets = ['haar', 'db2', 'sym2']  # You can add more wavelets if needed

for wavelet in wavelets:
    print(f'Using {wavelet} wavelet:')
    
    for i in range(3):  # Iterate over each color channel (0: Blue, 1: Green, 2: Red)
        # Apply wavelet transformation to the original and extracted channels
        coeffs_original = pywt.wavedec2(original_channels[i], wavelet, level=1)
        coeffs_extracted = pywt.wavedec2(extracted_channels[i], wavelet, level=1)

        # Calculate the MSE for each channel
        mse = np.mean((coeffs_original[0] - coeffs_extracted[0]) ** 2)
        
        # Print the MSE value for the current channel and wavelet
        channel_name = ['Blue', 'Green', 'Red'][i]
        print(f'MSE ({channel_name}): {mse}')
    
    print()
