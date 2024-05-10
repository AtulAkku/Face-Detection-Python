import pywt
import cv2
import numpy as np

# Load the watermarked image
watermarked_image = cv2.imread('watermarked_image.jpg', cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# Choose the same wavelet and decomposition level used for embedding
wavelet = 'sym4'
level = 2

# Perform the DWT transformation on the watermarked image
coeffs = pywt.wavedec2(watermarked_image, wavelet, level=level)

# Retrieve the watermarked image from the DWT coefficients
alpha = 0.2  # The same strength of the watermark used for embedding
watermarked_image_reconstructed = coeffs[0] / (1 + alpha)  # Corrected calculation

# Save the extracted watermarked image
cv2.imwrite('extracted_watermarked_image.jpg', watermarked_image_reconstructed)
