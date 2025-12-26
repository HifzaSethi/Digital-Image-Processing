import cv2
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread("F:\python\AI\dip\Week5\Adaptive.jpg", 0)

# Apply Adaptive Thresholding
adaptive = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# Display
plt.imshow(img)
plt.title("Orignal image")
plt.axis("off")
plt.show()
plt.imshow(adaptive, cmap='gray')
plt.title("Adaptive Thresholding")
plt.axis("off")
plt.show()
