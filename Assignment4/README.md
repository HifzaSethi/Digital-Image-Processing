# This ASSIGNMENT contain
# 1. Synthetic Image Creation and Thresholding
- **Objective**: Learn different thresholding techniques for image segmentation.
- **Steps:**
  1. Created a synthetic grayscale image of size 256×256 with:
     - Gradient background
     - Bright and dark regions using `cv2.circle`, `cv2.rectangle`, and `cv2.ellipse`
     - Added small noise for realism
  2. Applied **three thresholding methods**:
     - **Global Thresholding:** Fixed threshold
     - **Otsu’s Thresholding:** Automatic threshold selection
     - **Adaptive Gaussian Thresholding:** Pixel-wise adaptive threshold
  3. Displayed original and thresholded images using `matplotlib`.

- **Outcome:** Learned the difference between global, Otsu, and adaptive thresholding methods.

---

# 2. Image Steganography (Message Hiding)
- **Objective:** Hide a text message in an image using **LSB (Least Significant Bit) method**.
- **Steps:**
  1. Created a simple grayscale image with text `"HELLO AI"`.
  2. Converted the message `"Hi friend! I finished my assignment :)"` into binary bits.
  3. Embedded the message bits into the **LSB of image pixels**.
  4. Saved and displayed the secret image.
  5. Recovered the hidden message to verify correctness.

- **Outcome:** Practiced **basic steganography** and learned LSB manipulation.

---

# 3. Adaptive Thresholding for Uneven Lighting
- **Objective:** Handle images with uneven illumination using adaptive thresholding.
- **Steps:**
  1. Created a synthetic image with horizontal brightness gradient and added text and shadow.
  2. Applied:
     - **Adaptive Mean Thresholding**
     - **Adaptive Gaussian Thresholding**
  3. Compared results to observe which method handles uneven lighting better.

- **Outcome:** Learned how adaptive thresholding improves segmentation for non-uniformly lit images.

---
