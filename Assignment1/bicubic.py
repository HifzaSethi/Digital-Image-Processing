import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img_path = r"F:\python\AI\dip\assignmnet1\image.jpg"
img = cv2.imread(img_path)

if img is None:
    print("Could not load image. Check the path:", img_path)
else:
    print("Image loaded successfully!")

    # Resize original to 256x256 for base image
    img = cv2.resize(img, (256, 256))
    print("Original image scaled to 256×256 (Base image).")

    # ----------------------------
    # 🔹 Manual Bicubic Interpolation Function
    # ----------------------------
    def cubic_weight(t):
        """Cubic weight function used in bicubic interpolation."""
        a = -0.5  # Common choice for bicubic (Catmull-Rom spline)
        abs_t = abs(t)
        if abs_t <= 1:
            return (a + 2) * abs_t**3 - (a + 3) * abs_t**2 + 1
        elif abs_t < 2:
            return a * abs_t**3 - 5*a * abs_t**2 + 8*a * abs_t - 4*a
        else:
            return 0

    def bicubic_interpolation(image, scale):
        h, w, c = image.shape
        new_h, new_w = int(h * scale), int(w * scale)
        new_img = np.zeros((new_h, new_w, c), dtype=np.uint8)

        for y_new in range(new_h):
            for x_new in range(new_w):
                x = x_new / scale
                y = y_new / scale

                x_int = int(np.floor(x))
                y_int = int(np.floor(y))

                for ch in range(c):
                    value = 0
                    total_weight = 0
                    # Consider 4x4 neighborhood
                    for m in range(-1, 3):
                        for n in range(-1, 3):
                            x_neigh = min(max(x_int + n, 0), w - 1)
                            y_neigh = min(max(y_int + m, 0), h - 1)

                            wx = cubic_weight(x - (x_int + n))
                            wy = cubic_weight(y - (y_int + m))
                            weight = wx * wy

                            total_weight += weight
                            value += image[y_neigh, x_neigh, ch] * weight

                    new_img[y_new, x_new, ch] = np.clip(value / total_weight, 0, 255)

        return new_img

    # ----------------------------
    # 🔹 Perform Manual Bicubic Upscaling
    # ----------------------------
    bicubic_2x = bicubic_interpolation(img, 2)
    bicubic_4x = bicubic_interpolation(img, 4)
    print("Manual Bicubic Interpolation done (×2, ×4).")

    # ----------------------------
    # 🔹 Mean Squared Error Function
    # ----------------------------
    def mse(imageA, imageB):
        return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

    # Resize original image to same size for comparison
    original_2x = cv2.resize(img, (bicubic_2x.shape[1], bicubic_2x.shape[0]), interpolation=cv2.INTER_CUBIC)
    original_4x = cv2.resize(img, (bicubic_4x.shape[1], bicubic_4x.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Compute MSE
    mse_2x = mse(original_2x, bicubic_2x)
    mse_4x = mse(original_4x, bicubic_4x)

    print("\nMean Squared Error (MSE) Results for Manual Bicubic Interpolation:")
    print(f"   ×2 Scaling → MSE = {mse_2x:.2f}")
    print(f"   ×4 Scaling → MSE = {mse_4x:.2f}")

    # ----------------------------
    # 🔹 Show Images
    # ----------------------------
    cv2.imshow("Original (256×256)", img)
    cv2.imshow("Bicubic ×2 (512×512)", bicubic_2x)
    cv2.imshow("Bicubic ×4 (1024×1024)", bicubic_4x)

    print("\nShowing images separately... (Press any key to continue)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ----------------------------
    # 🔹 Plot in Matplotlib
    # ----------------------------
    images = [img, bicubic_2x, bicubic_4x]
    titles = ["Original (256×256)", "Bicubic ×2 (512×512)", "Bicubic ×4 (1024×1024)"]

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
