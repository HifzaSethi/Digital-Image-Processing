import cv2
import numpy as np
import matplotlib.pyplot as plt


#create own image
h, w = 256, 256
img = np.zeros((h, w), dtype=np.uint8)

# Create gradient background
for i in range(h):
    img[i, :] = np.linspace(40, 220, w)

# Add bright and dark regions
cv2.circle(img, (80, 100), 40, 30, -1)        # Dark circle
cv2.rectangle(img, (140, 40), (230, 120), 200, -1)  # Bright rectangle
cv2.ellipse(img, (160, 180), (60, 30), 0, 0, 360, 120, -1)  # Mid-bright ellipse

# Add small noise for realism
noise = (np.random.randn(h, w) * 10).astype(np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Show original image
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

# ---------------- Step 2: Apply THREE types of Thresholding ----------------

# 1. Global Thresholding (Fixed threshold value)
thresh_value = 127
_, th_global = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)

# 2. Otsu’s Thresholding (Automatic)
_, th_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. Adaptive Gaussian Thresholding
th_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, blockSize=31, C=8)

# ---------------- Step 3: Display all results ----------------
titles = ['Original Image', 
          f'Global Thresholding (T={thresh_value})', 
          f"Otsu’s Thresholding (T={int(_):d})", 
          'Adaptive Gaussian Thresholding']
images = [img, th_global, th_otsu, th_adaptive]

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
