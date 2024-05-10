import cv2
import numpy as np
import pywt

# Load the original and watermarked images
original_image = cv2.imread('your-image~1.jpg', cv2.IMREAD_COLOR)
watermarked_image = cv2.imread('watermarked_image.jpg', cv2.IMREAD_COLOR)

# Ensure both images have the same dimensions
original_image = cv2.resize(original_image,(watermarked_image.shape[1], watermarked_image.shape[0]))

# Split the images into their RGB channels
original_channels = cv2.split(original_image)
watermarked_channels = cv2.split(watermarked_image)

wavelets = ['haar', 'db2', 'sym2']  # You can add more wavelets if needed

for wavelet in wavelets:
    print(f'Using {wavelet} wavelet:')
    
    for i in range(3):  # Iterate over each color channel (0: Blue, 1: Green, 2: Red)
        # Apply wavelet transformation
        coeffs_original = pywt.wavedec2(original_channels[i], wavelet, level=1)
        coeffs_watermarked = pywt.wavedec2(watermarked_channels[i], wavelet, level=1)

        # Calculate the PSNR for each channel
        mse = np.mean((coeffs_original[0] - coeffs_watermarked[0]) ** 2)
        max_pixel_value = 255  # Assuming the images have pixel values in the range [0, 255]
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        
        # Print the PSNR value for the current channel and wavelet
        channel_name = ['Blue', 'Green', 'Red'][i]
        print(f'PSNR ({channel_name}): {psnr} dB')
    
    print()
