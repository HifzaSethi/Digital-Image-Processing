import cv2
import numpy as np

# Load the image
image = cv2.imread('F:\python\AI\dip\Week5\Traditional\contour.jpg')  
cv2.imshow("Original Image", image)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours from the edges
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image to draw contours
contour_image = np.zeros_like(image)

# Draw contours on the blank image
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Show contour image
cv2.imshow("Contour Extracted Image", contour_image)

# Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
