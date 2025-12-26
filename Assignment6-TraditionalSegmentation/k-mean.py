import cv2
import numpy as np

# Use raw string or forward slashes
image = cv2.imread(r'F:\python\AI\dip\Week5\Traditional\kMean.jpg')

if image is None:
    print("Error: Image not found. Check the path!")
    exit() 

cv2.imshow("Original Image", image)
cv2.waitKey(0)

# K-means clustering
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4

_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(image.shape)

cv2.imshow("K-means Clustered Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
