import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread(r"F:\python\AI\dip\Week5\Traditional\otsu's.jpg", 0)

# Check if image loaded successfully
if img is None:
    raise FileNotFoundError("Image not found. Check the file path!")

# Apply Otsu's Thresholding
threshold, binary = cv2.threshold(
    img,
    0,                  # threshold ignored
    255,                # max value
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Display Original Image first
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Display Otsu Thresholded Image
plt.subplot(1,2,2)
plt.imshow(binary, cmap='gray')
plt.title(f"Otsu Thresholded (T={threshold:.0f})")
plt.axis("off")

plt.show()
