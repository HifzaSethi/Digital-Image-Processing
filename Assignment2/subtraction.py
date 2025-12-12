import cv2
import numpy as np
import matplotlib.pyplot as plt

# 🔹 Load two images (for example, original and modified/cropped part)
img1_path = r"F:\python\AI\dip\task\normal.jpg"   # first cropped half
img2_path = r"F:\python\AI\dip\task\tumor.jpg"   # second cropped half

# 🔹 Read images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 🔹 Check if loaded
if img1 is None or img2 is None:
    print("Error: Could not load one or both images.")
    exit()

# 🔹 Resize both to same size for arithmetic operations
img1 = cv2.resize(img1, (256, 256))
img2 = cv2.resize(img2, (256, 256))

# -------------------------------------------------------------
# 🟣 1. IMAGE SUBTRACTION
# -------------------------------------------------------------
# Subtracts pixel values to find difference between the two
subtracted = cv2.subtract(img1, img2)

# -------------------------------------------------------------
# 🟢 2. IMAGE MULTIPLICATION
# -------------------------------------------------------------
# Multiplies pixel intensities → emphasizes overlapping bright areas
# Normalize result to keep within valid range
multiplied = cv2.multiply(img1, img2)
multiplied = cv2.normalize(multiplied, None, 0, 255, cv2.NORM_MINMAX)
multiplied = multiplied.astype(np.uint8)

# -------------------------------------------------------------
# 🔹 Display Results
# -------------------------------------------------------------
titles = ['Image 1', 'Image 2', 'Subtraction', 'Multiplication']
images = [img1, img2, subtracted, multiplied]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 🔹 Save results (optional)
# -------------------------------------------------------------
cv2.imwrite(r"F:\python\AI\dip\task\subtracted.jpg", subtracted)
cv2.imwrite(r"F:\python\AI\dip\task\multiplied.jpg", multiplied)

print("✅ Image subtraction and multiplication done successfully!")
