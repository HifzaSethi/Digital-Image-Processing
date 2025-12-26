import cv2
import numpy as np

# Step 1: Read the image
image = cv2.imread('F:\python\AI\dip\Week5\Traditional\watershed.png')  
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Otsu's thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 4: Noise removal with morphological opening
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 5: Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Step 6: Sure foreground area using distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Step 7: Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# Step 8: Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown==255] = 0

# Step 9: Apply watershed
markers = cv2.watershed(image, markers)

# Step 10: Mark boundaries in red
image[markers == -1] = [0,0,255]

# Step 11: Show watershed result
cv2.imshow("Watershed Edge Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
