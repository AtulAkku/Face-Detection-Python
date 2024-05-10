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

# Retrieve the watermark from the low-frequency component (same location where it was embedded)
alpha = 0.1  # Adjust the same strength of the watermark used for embedding
watermark_gray = (coeffs[0] / alpha).astype(np.uint8)  # Corrected calculation

# Resize the watermark to its original size (if needed)
# watermark_gray = cv2.resize(watermark_gray, (watermark_width, watermark_height))

# Save the extracted watermark
cv2.imwrite('extracted_watermark.png', watermark_gray)
