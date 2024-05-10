import cv2
import numpy as np

# Load the original and watermarked images
original_image = cv2.imread('your-image~1.jpg', cv2.IMREAD_COLOR)
watermarked_image = cv2.imread('watermarked_image.jpg', cv2.IMREAD_COLOR)

# Ensure both images have the same dimensions
original_image = cv2.resize(original_image, (watermarked_image.shape[1], watermarked_image.shape[0]))

# Split the images into their RGB channels
original_channels = cv2.split(original_image)
watermarked_channels = cv2.split(watermarked_image)

psnr_values = []

for i in range(3):  # Iterate over each color channel (0: Blue, 1: Green, 2: Red)
    mse = np.mean((original_channels[i] - watermarked_channels[i]) ** 2)
    max_pixel_value = 255  # Assuming the images have pixel values in the range [0, 255]
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    psnr_values.append(psnr)

# PSNR values for each color channel
psnr_blue, psnr_green, psnr_red = psnr_values

print(f'PSNR (Blue): {psnr_blue} dB')
print(f'PSNR (Green): {psnr_green} dB')
print(f'PSNR (Red): {psnr_red} dB')
