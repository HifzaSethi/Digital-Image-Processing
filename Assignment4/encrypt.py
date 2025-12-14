import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Step 1: Create a simple grayscale image ----------
# 100 rows × 300 columns, white background (255)
img = np.ones((100, 300), dtype=np.uint8) * 255

# Add some black text on it
cv2.putText(img, "HELLO AI", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0), 3)

# ---------- Step 2: Message to hide ----------
message = "Hi friend! I finished my assignment :)"
bits = ''.join([format(ord(c), '08b') for c in message])  # convert each letter to 8-bit binary

# Make a copy of image for hiding
secret_img = img.copy().flatten()  # flatten to 1D for easy bit manipulation

# Hide message bits in the last bit (LSB) of each pixel
for i in range(len(bits)):
    secret_img[i] = (secret_img[i] & 254) | int(bits[i])

secret_img = secret_img.reshape(img.shape)  # back to 2D shape

# ---------- Step 3: Save and display both ----------
cv2.imwrite("F:/python/AI/dip/task3/secret_image.png", secret_img)

# Show original and secret image side by side
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(secret_img, cmap='gray')
plt.title("Secret (Encrypted) Image")
plt.axis('off')
plt.show()

# ---------- Step 4: Decode the message ----------
flat = secret_img.flatten()
recovered_bits = [str(flat[i] & 1) for i in range(len(bits))]
recovered_message = ''.join([chr(int(''.join(recovered_bits[i:i+8]), 2)) for i in range(0, len(recovered_bits), 8)])

print("Hidden Message:", recovered_message)
