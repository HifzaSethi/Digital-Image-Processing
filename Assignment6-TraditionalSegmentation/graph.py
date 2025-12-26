import cv2
import numpy as np

# Step 1: Read the image
image = cv2.imread(r'F:\python\AI\dip\Week5\Traditional\graph.jpg')  # Replace with your image path

if image is None:
    print("Error: Image not found. Check the path!")
    exit()

cv2.imshow("Original Image", image)
cv2.waitKey(0)

# Step 2: Create mask and models for grabCut
mask = np.zeros(image.shape[:2], np.uint8)  # Mask initialized to 0
bgdModel = np.zeros((1,65), np.float64)     # Background model
fgdModel = np.zeros((1,65), np.float64)     # Foreground model

# Step 3: Define rectangle around foreground (x, y, width, height)
# Adjust the rectangle to enclose your main object
rect = (50, 50, image.shape[1]-100, image.shape[0]-100)  

# Step 4: Apply grabCut
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Step 5: Prepare final mask
# Pixels marked as sure foreground or probable foreground are set to 1, others to 0
mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
segmented_image = image * mask2[:, :, np.newaxis]  # Apply mask to original image

# Step 6: Show result
cv2.imshow("Graph Cut Segmentation", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
