import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.segmentation import chan_vese
from skimage.filters import gaussian


# Step 1: Load image

image_path = r"F:\python\AI\dip\Week5\classical\level.jpg"  # Replace with your image path
image = io.imread(image_path)

# Convert to grayscale and float
gray = color.rgb2gray(image)
gray = img_as_float(gray)

# Optional: smooth image to reduce noise
gray_smooth = gaussian(gray, sigma=1)


# Step 2: Apply Chan-Vese Level Set Segmentation

# For older versions, use only supported parameters
cv_result = chan_vese(
    gray_smooth,
    mu=0.25,        # smoothness parameter
    lambda1=1,      # weight for inside intensity
    lambda2=1,      # weight for outside intensity
    tol=1e-3,       # convergence tolerance
    dt=0.5,         # time step
    init_level_set="checkerboard"  # initial mask
)


# Step 3: Show Results

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(gray, cmap='gray')
axes[1].contour(cv_result, [0.5], colors='r')  # contour overlay
axes[1].set_title("Chan-Vese Segmentation")
axes[1].axis("off")

plt.tight_layout()
plt.show()
