import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image (use grayscale)
image = cv2.imread("F:\python\AI\dip\Week5\prewitt.png", cv2.IMREAD_GRAYSCALE)

# Prewitt kernels
kernel_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

kernel_y = np.array([
    [ 1,  1,  1],
    [ 0,  0,  0],
    [-1, -1, -1]
])

# Apply convolution
prewitt_x = cv2.filter2D(image, -1, kernel_x)
prewitt_y = cv2.filter2D(image, -1, kernel_y)

# Edge magnitude
prewitt_edges = np.sqrt(prewitt_x**2 + prewitt_y**2)
prewitt_edges = np.uint8(prewitt_edges)

# Display results
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Prewitt X")
plt.imshow(prewitt_x, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Prewitt Edges")
plt.imshow(prewitt_edges, cmap='gray')
plt.axis('off')

plt.show()
