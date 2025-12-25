import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import warnings


# Step 1: Load image

image_path = r"F:\python\AI\dip\Week5\classical\active.jpg"
image = io.imread(image_path)

# Convert to grayscale
gray = color.rgb2gray(image)

# Smooth image to reduce noise
gray_smooth = gaussian(gray, sigma=1)


# Step 2: Thresholding to roughly segment objects
thresh = filters.threshold_otsu(gray_smooth)
binary = gray_smooth < thresh  # Objects are True

# Remove small objects (suppress future warning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    binary = morphology.remove_small_objects(binary, min_size=500)


# Step 3: Find contours

contours = measure.find_contours(binary, level=0.8)

# Step 4: Apply Active Contour (Snake) to each contour

snake_results = []

for contour in contours:
    # Subsample if contour has too many points
    if len(contour) > 200:
        idx = np.linspace(0, len(contour) - 1, 200).astype(int)
        init = contour[idx]
    else:
        init = contour

    # Apply active contour
    snake = active_contour(
        gray_smooth,
        init,
        alpha=0.015,
        beta=10,
        gamma=0.001
    )
    snake_results.append(snake)


# Step 5: Show original image
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.show()


# Step 6: Show image with snakes

plt.figure(figsize=(8, 8))
plt.imshow(image)
for snake in snake_results:
    plt.plot(snake[:, 1], snake[:, 0], '-b', lw=2)
plt.title("Active Contours on Multiple Objects")
plt.axis("off")
plt.show()

