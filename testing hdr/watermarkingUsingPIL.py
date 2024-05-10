from PIL import Image

# Load the original image
original_image = Image.open('your-image~1.jpg')

# Load the watermark image (with transparency)
watermark = Image.open('watermark.png')

# You may need to resize the watermark image to fit your original image
# Uncomment the following line and adjust the size as needed
# watermark = watermark.resize((width, height))

# Create a copy of the original image to avoid modifying it directly
image_with_watermark = original_image.copy()

# Position the watermark on the bottom right corner (you can adjust the position)
position = (original_image.width - watermark.width, original_image.height - watermark.height)

# Composite the watermark onto the original image
image_with_watermark = Image.alpha_composite(image_with_watermark.convert('RGBA'), watermark.convert('RGBA'))

# Save the image with the watermark
image_with_watermark.save('image_with_watermark.png')

# Show the watermarked image (optional)
image_with_watermark.show()
