import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load  image
img_path = r"F:\python\AI\dip\assignmnet1\image.jpg"
img = cv2.imread(img_path)

if img is None:
    print("Could not load image. Check path:", img_path)
else:
    print("Image loaded successfully!")

    # Step 1: Resize manually to 256x256 (for base image)
    img = cv2.resize(img, (256, 256))
    print("Base image resized to 256×256 for consistency.")

    #  Manual Nearest Neighbor Interpolation 
    def nearest_neighbor_interpolation(image, scale):
        #hight,wiidth,color
        h, w, c = image.shape
        new_h, new_w = int(h * scale), int(w * scale)
        new_img = np.zeros((new_h, new_w, c), dtype=np.uint8)

        for y_new in range(new_h):
            for x_new in range(new_w):
                # Map back to original coordinates
                y_old = int(round(y_new / scale))
                x_old = int(round(x_new / scale))

                # Clamp to valid indices
                y_old = min(y_old, h - 1)
                x_old = min(x_old, w - 1)

                new_img[y_new, x_new] = image[y_old, x_old]

        return new_img

    # Perform manual upscaling
    nearest_2x = nearest_neighbor_interpolation(img, 2)
    nearest_4x = nearest_neighbor_interpolation(img, 4)
    print("Manual nearest neighbor interpolation done (×2, ×4).")

    # Compute Mean Squared Error
    def mse(imageA, imageB):
        # Ensure same size
        if imageA.shape != imageB.shape:
            imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))
        return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

    mse_2x = mse(cv2.resize(img, (nearest_2x.shape[1], nearest_2x.shape[0])), nearest_2x)
    mse_4x = mse(cv2.resize(img, (nearest_4x.shape[1], nearest_4x.shape[0])), nearest_4x)

    print("\nMean Squared Error (MSE) Results:")
    print(f"   ×2 Scaling → MSE = {mse_2x:.2f}")
    print(f"   ×4 Scaling → MSE = {mse_4x:.2f}")

    #  Show Images Using OpenCV 
    cv2.imshow("Original (256×256)", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Nearest Neighbor ×2 (512×512)", cv2.cvtColor(nearest_2x, cv2.COLOR_RGB2BGR))
    cv2.imshow("Nearest Neighbor ×4 (1024×1024)", cv2.cvtColor(nearest_4x, cv2.COLOR_RGB2BGR))
    print("\nShowing images separately... (Press any key to continue)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Show Images Using Matplotlib 
    images = [img, nearest_2x, nearest_4x]
    titles = ["Original (256×256)", "Nearest Neighbor ×2 (512×512)", "Nearest Neighbor ×4 (1024×1024)"]

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
