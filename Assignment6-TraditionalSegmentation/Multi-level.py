import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread("F:\python\AI\dip\Week5\multi-level.jpg", 0)

# Define thresholds
T1 = 80
T2 = 160

# Create output image
output = np.zeros_like(img)

# Apply multi-level thresholding
output[img < T1] = 50
output[(img >= T1) & (img < T2)] = 150
output[img >= T2] = 255

# Display
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")
plt.show()
plt.imshow(output, cmap='gray')
plt.title("Multi-Level Thresholding")
plt.axis("off")
plt.show()
