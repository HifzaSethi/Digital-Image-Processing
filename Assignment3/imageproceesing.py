import cv2
import numpy as np
import matplotlib.pyplot as plt

#Step 1: Create grayscale test image 

img = cv2.imread(r"F:\python\AI\dip\task2\xray.jpg", 0)

# Step 2: Image Negative 
L = 256
negative = L - 1 - img

#  Step 3: Contrast Stretching ----------
r1, s1 = 70, 0
r2, s2 = 140, 255

def contrast_stretching(r):
    # Apply piecewise function
    if r < r1:
        return (s1 / r1) * r
    elif r < r2:
        return ((s2 - s1) / (r2 - r1)) * (r - r1) + s1
    else:
        return ((L-1 - s2) / (L-1 - r2)) * (r - r2) + s2

# Vectorize function for all pixels
contrast = np.vectorize(contrast_stretching)(img).astype(np.uint8)

# Step 4: Show Results
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(negative, cmap='gray')
plt.title('Image Negative')

plt.subplot(1,3,3)
plt.imshow(contrast, cmap='gray')
plt.title('Contrast Stretched')

plt.show()
