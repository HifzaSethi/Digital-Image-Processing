import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Step 1: Create a meaningful synthetic image ----------
img = np.zeros((200, 400), dtype=np.uint8)

# Create a gradient (left dark, right bright)
for i in range(img.shape[1]):
    img[:, i] = i // 2  # brightness increases horizontally

# Add text (simulate object)
cv2.putText(img, "HELLO AI", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 5, cv2.LINE_AA)

# Add a shadow on the top-left region
cv2.rectangle(img, (0, 0), (150, 100), (50), -1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Uneven Lighting Image")
plt.axis("off")

# ---------- Step 2: Apply Adaptive Mean Thresholding ----------
th_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 21, 10)

# ---------- Step 3: Apply Adaptive Gaussian Thresholding ----------
th_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 10)

# ---------- Step 4: Display Results ----------
plt.subplot(1, 3, 2)
plt.imshow(th_mean, cmap='gray')
plt.title("Adaptive Mean Thresholding")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(th_gaussian, cmap='gray')
plt.title("Adaptive Gaussian Thresholding")
plt.axis("off")

plt.show()
