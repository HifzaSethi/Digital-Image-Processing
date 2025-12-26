import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read image in grayscale (cv2 handles it)
img = cv2.imread(r"F:\python\AI\dip\Week5\Traditional\Canny.jpg", 0)

# Check if image loaded successfully
if img is None:
    raise FileNotFoundError("Image not found. Check the file path!")

# Apply Canny Edge Detection
edges = cv2.Canny(img, 50, 150)

# Display Original
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")
plt.show()

# Display Edges
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis("off")
plt.show()
