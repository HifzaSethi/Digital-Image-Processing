import cv2
import numpy as np

# ---------- Step 1: Read both images ----------
# Use raw strings (r"...") or double backslashes \\ to avoid path errors
img1 = cv2.imread(r"F:\python\AI\dip\task\image1.jpg")   # darker image
img2 = cv2.imread(r"F:\python\AI\dip\task\image2.jpg")   # brighter image

# Check if images loaded correctly
if img1 is None or img2 is None:
    print("❌ Error: One or both image paths are incorrect. Check file names!")
    exit()

# ---------- Step 2: Resize to same shape ----------
# Both images must be same width and height for pixel-wise operations
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# ---------- Step 3: Add both images ----------
# Each pixel from img1 and img2 is added together.
# If sum > 255, it is clipped (set) to 255 (max brightness).
added_img = cv2.add(img1, img2)

# ---------- Step 4: Display results ----------
cv2.imshow("Image 1 (Dark)", img1)
cv2.imshow("Image 2 (Bright)", img2)
cv2.imshow("Added Image (Result)", added_img)

# ---------- Step 5: Save result ----------
cv2.imwrite(r"F:\python\AI\dip\task\added_result.jpg", added_img)
print("✅ Added image saved successfully as added_result.jpg")

#Step 6: Wait for user 
cv2.waitKey(0)
cv2.destroyAllWindows()
