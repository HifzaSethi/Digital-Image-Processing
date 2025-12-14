import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter, binary_opening, binary_closing
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes

# 1. LOAD PBM IMAGE

img = Image.open(r"F:\python\AI\dip\taskweek3\images lab\images lab\testHard.pbm")
img = np.array(img)

# PBM images are 0/1; convert to binary
binary_img = (img > 0).astype(np.uint8)


# 2. NOISE REMOVAL (Median)

denoised = median_filter(binary_img, size=3)


# 3. MORPHOLOGICAL OPENING (remove small white noise)

opened = binary_opening(denoised, structure=np.ones((3,3)))

# 4. MORPHOLOGICAL CLOSING (close gaps & holes)

closed = binary_closing(opened, structure=np.ones((3,3)))

# 5. REMOVE BRIDGES & BRANCHES
# (Skeletonize + remove small branches)


# Skeleton of the object
skeleton = skeletonize(closed)

# Remove very small objects from skeleton
skeleton_clean = remove_small_objects(skeleton, min_size=20)

# Optional: Remove small holes inside object
clean_img = remove_small_holes(skeleton_clean, area_threshold=30)


# 6. NORMALIZATION (for visualization)

img_float = closed.astype(np.float32)
min_val = np.min(img_float)
max_val = np.max(img_float)
normalized = (img_float - min_val) / (max_val - min_val)

print("\n--- Normalization ---")
print(f"Min pixel value = {min_val}")
print(f"Max pixel value = {max_val}")


# 7. EXTRACT MAIN OBJECT MASK (threshold)

threshold = 0.5
object_mask = (normalized < threshold).astype(np.uint8)

# Remove small isolated objects
object_mask = remove_small_objects(object_mask.astype(bool), min_size=100).astype(np.uint8)


# 8. GET OBJECT COORDINATES

ys, xs = np.where(object_mask == 1)
min_x, max_x = np.min(xs), np.max(xs)
min_y, max_y = np.min(ys), np.max(ys)

print("\n--- Object Coordinates ---")
print(f"Min X = {min_x}, Max X = {max_x}")
print(f"Min Y = {min_y}, Max Y = {max_y}")


# 9. CREATE BOUNDING BOX IMAGE

bbox_img = np.stack([object_mask*255]*3, axis=2)

# Draw box
bbox_img[min_y:max_y, min_x] = [255, 0, 0]   # left
bbox_img[min_y:max_y, max_x] = [255, 0, 0]   # right
bbox_img[min_y, min_x:max_x] = [255, 0, 0]   # top
bbox_img[max_y, min_x:max_x] = [255, 0, 0]   # bottom


# 10. DISPLAY RESULTS

plt.figure(figsize=(18,8))

plt.subplot(2,5,1)
plt.imshow(img, cmap="gray")
plt.title("Original PBM")
plt.axis("off")

plt.subplot(2,5,2)
plt.imshow(denoised, cmap="gray")
plt.title("Median Filter")
plt.axis("off")

plt.subplot(2,5,3)
plt.imshow(opened, cmap="gray")
plt.title("Opening")
plt.axis("off")

plt.subplot(2,5,4)
plt.imshow(closed, cmap="gray")
plt.title("Closing")
plt.axis("off")

plt.subplot(2,5,5)
plt.imshow(skeleton_clean, cmap="gray")
plt.title("Clean Skeleton (Bridges Removed)")
plt.axis("off")

plt.subplot(2,5,6)
plt.imshow(normalized, cmap="gray")
plt.title("Normalized")
plt.axis("off")

plt.subplot(2,5,7)
plt.imshow(object_mask, cmap="gray")
plt.title("Object Mask")
plt.axis("off")

plt.subplot(2,5,8)
plt.imshow(bbox_img)
plt.title("Bounding Box")
plt.axis("off")

plt.tight_layout()
plt.show()
