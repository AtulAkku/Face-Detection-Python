import cv2
import numpy as np

# Initialize variables
m = 256  # Size of Imgra (m X m)
n = 64   # Size of wmr (n X n)
L = m * m
Z = 64  # Number of blocks for embimg1
watermark_bits = [0, 1, 0, 1, 1, 0, 1, 0]  # Example watermark bits

# Read the HDR image (I)
hdr_image = cv2.imread('your-image~1.jpg')

# Ensure the HDR image is in the correct format (CV_32FC3)
hdr_image = hdr_image.astype(np.float32) / 255.0  # Normalize to the range [0, 1]

# Read the watermark image (Wtg)
watermark_image = cv2.imread('watermark.png')

# Tonemap the HDR image (Timage)
tonemapped_image = cv2.createTonemapMantiuk().process(hdr_image)

# Convert the tonemapped image to grayscale (Imgrey)
gray_image = cv2.cvtColor(tonemapped_image, cv2.COLOR_BGR2GRAY)

# Resize the grayscale image to m X m (Imgra)
Imgra = cv2.resize(gray_image, (m, m))

# Resize the watermark image to n X n (wmr)
wmr = cv2.resize(watermark_image, (n, n))

# Block-wise DCT for Imgra
x = []
for i in range(0, m, 8):
    for j in range(0, m, 8):
        block = Imgra[i:i+8, j:j+8]
        dct_block = cv2.dct(block.astype(np.float32))
        x.append(dct_block)

# Watermark embedding
w = 0
for k in range(min(L, len(x))):
    kx = x[k].copy()
    for i in range(8):  # Iterate only up to 8
        for j in range(8):  # Iterate only up to 8
            if i == 7 and j == 7 and w < len(watermark_bits):
                if watermark_bits[w] == 0:
                    kx[i, j] = kx[i, j] + 35
                elif watermark_bits[w] == 1:
                    kx[i, j] = kx[i, j] - 35
                w += 1
    x[k] = kx

# Reconstruct embimg1
embimg1 = []
count = 0
for j in range(0, min(L, len(x)), 64):  # Ensure the loop doesn't go out of bounds
    data = []
    for i in range (j, min(j + 64, min(L, len(x)))):
        data.extend(x[i].flatten())
    embimg1.append(data)
    count += 1

# Concatenate embimg1 to get dembimg
dembimg = np.concatenate(embimg1, axis=0)

# Inverse DCT on dembimg
dct_blocks = np.split(dembimg, L)
for i in range(L):
    dct_block = cv2.idct(dct_blocks[i])
    dembimg[i] = dct_block[0, 0]  # Extract a single element from the DCT block


# Now, dembimg contains the watermarked image

# You can save or display the watermarked image as needed
# cv2.imwrite('watermarked_image.jpg', dembimg)

# Specify the full path for saving the watermarked image
output_path = 'C:/Users/Atul_Akku/Desktop/face recognation/testing hdr/watermarked_image.jpg'

# Save the watermarked image
cv2.imwrite(output_path, dembimg)
