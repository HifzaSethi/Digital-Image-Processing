import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img_path = r"F:\python\AI\dip\assignmnet1\image.jpg"
img = cv2.imread(img_path)

if img is None:
    print("Could not load image. Check path:", img_path)
else:
    print("Image loaded successfully!")

    # Convert to RGB (for plotting)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize original to 256x256 (for base image only, allowed)
    img = cv2.resize(img, (256, 256))
    print("Original image scaled to 256×256 (Base image).")

 
    # Manual Bilinear Interpolation Function

    def bilinear_interpolation(image, scale):
        h, w, c = image.shape
        new_h, new_w = int(h * scale), int(w * scale)
        new_img = np.zeros((new_h, new_w, c), dtype=np.uint8)

        for y_new in range(new_h):
            for x_new in range(new_w):
                # Map to original coordinates
                x = x_new / scale
                y = y_new / scale

                # Coordinates of surrounding pixels
                x1 = int(np.floor(x))
                y1 = int(np.floor(y))
                x2 = min(x1 + 1, w - 1)
                y2 = min(y1 + 1, h - 1)

                # Fractional parts
                dx = x - x1
                dy = y - y1

                # For each color channel
                for ch in range(c):
                    Q11 = image[y1, x1, ch]
                    Q21 = image[y1, x2, ch]
                    Q12 = image[y2, x1, ch]
                    Q22 = image[y2, x2, ch]

                    # Bilinear formula
                    value = (Q11 * (1 - dx) * (1 - dy) +
                             Q21 * dx * (1 - dy) +
                             Q12 * (1 - dx) * dy +
                             Q22 * dx * dy)

                    new_img[y_new, x_new, ch] = int(value)

        return new_img

   
    #  Perform Manual Bilinear Upscaling
    bilinear_2x = bilinear_interpolation(img, 2)
    bilinear_4x = bilinear_interpolation(img, 4)
    print("Manual Bilinear Interpolation done (×2, ×4).")


    #  Manual Mean Squared Error Function
   
    def mse(imageA, imageB):
        h, w, c = imageA.shape
        total_error = 0
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    diff = int(imageA[i, j, k]) - int(imageB[i, j, k])
                    total_error += diff * diff
        mse_value = total_error / (h * w * c)
        return mse_value

    # To compare, downscale the upscaled images back to 256×256 manually (fair test)
    down_2x = bilinear_interpolation(img, 2)
    down_2x = cv2.resize(down_2x, (256, 256))  # only for comparison dimension
    down_4x = bilinear_interpolation(img, 4)
    down_4x = cv2.resize(down_4x, (256, 256))

    # Compute MSE between original and downscaled versions
    mse_2x = mse(img, down_2x)
    mse_4x = mse(img, down_4x)

    print("\nMean Squared Error (MSE) Results for Manual Bilinear Interpolation:")
    print(f"   ×2 Scaling → MSE = {mse_2x:.2f}")
    print(f"   ×4 Scaling → MSE = {mse_4x:.2f}")

    #  Display Images Using OpenCV
   
    cv2.imshow("Original (256×256)", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Bilinear ×2 (512×512)", cv2.cvtColor(bilinear_2x, cv2.COLOR_RGB2BGR))
    cv2.imshow("Bilinear ×4 (1024×1024)", cv2.cvtColor(bilinear_4x, cv2.COLOR_RGB2BGR))

    print("\nShowing images separately... (Press any key to continue)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #  Display in Matplotlib
 
    images = [img, bilinear_2x, bilinear_4x]
    titles = [
        "Original (256×256)",
        "Bilinear ×2 (512×512)",
        "Bilinear ×4 (1024×1024)"
    ]

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
