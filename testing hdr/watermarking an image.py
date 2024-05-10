import pywt
import numpy as np
import cv2

# Load the HDR image
hdr_image = cv2.imread('your-image~1.jpg', cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# Choose a Symlet wavelet and decomposition level
wavelet = 'sym4'
level = 2

# Perform the DWT transformation
coeffs = pywt.wavedec2(hdr_image, wavelet, level=level)

# Select a watermark image
watermark = cv2.imread('watermark.png', cv2.IMREAD_COLOR)
watermark_gray = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)

# Resize the watermark to match the low-frequency component
watermark_gray = cv2.resize(watermark_gray, coeffs[0].shape[::-1])

# Embed the watermark into the DWT coefficients (add it to the low-frequency component)
alpha = 0.2  # Adjust the strength of the watermark
coeffs[0] += alpha * watermark_gray

# Perform the inverse DWT transformation
watermarked_image = pywt.waverec2(coeffs, wavelet)

# Save the watermarked image
cv2.imwrite('watermarked_image.jpg', watermarked_image)
