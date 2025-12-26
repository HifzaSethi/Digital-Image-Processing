# Assignment 6: Traditional Image Segmentation Methods

## Overview

This assignment demonstrates the implementation of various **traditional image segmentation techniques** using Python and OpenCV. The goal is to understand and apply fundamental image processing algorithms to extract meaningful information from images, including edges, regions, and objects.

## Segmentation Techniques Implemented

1. **Adaptive Thresholding**

   - Converts a grayscale image to a binary image based on the local neighborhood of pixels.
   - Useful for images with varying lighting conditions.
   - **Implementation:** `cv2.adaptiveThreshold`

2. **Canny Edge Detection**

   - Detects edges by identifying areas of rapid intensity change.
   - Includes noise reduction via Gaussian blur and hysteresis thresholding.
   - **Implementation:** `cv2.Canny`

3. **Contour Extraction**

   - Identifies and draws the boundaries of objects in an image.
   - Uses Canny edges as input for contour detection.
   - **Implementation:** `cv2.findContours` and `cv2.drawContours`

4. **Graph Cut Segmentation (GrabCut)**

   - Segments the foreground from the background using iterative graph cut optimization.
   - Requires an initial rectangle around the object of interest.
   - **Implementation:** `cv2.grabCut`

5. **K-means Clustering**

   - Segments the image by clustering similar pixel colors.
   - Reduces the number of colors to `k` clusters for segmentation.
   - **Implementation:** `cv2.kmeans`

6. **Multi-level Thresholding**

   - Divides an image into multiple intensity levels using predefined thresholds.
   - Provides a more nuanced segmentation than simple binary thresholding.
   - **Implementation:** NumPy-based thresholding

7. **Normalized Cut Segmentation**

   - Performs superpixel segmentation (SLIC) followed by graph-based normalized cut.
   - Segments an image based on color similarity and region adjacency.
   - **Implementation:** `skimage.segmentation.slic` and `skimage.graph.cut_normalized`

8. **Otsu's Thresholding**

   - Automatically determines the optimal threshold to separate foreground and background.
   - Particularly effective for bimodal intensity histograms.
   - **Implementation:** `cv2.threshold` with `cv2.THRESH_OTSU`

9. **Prewitt Edge Detection**

   - Detects edges using gradient operators along horizontal and vertical directions.
   - Highlights edge magnitude using the combination of X and Y gradients.
   - **Implementation:** `cv2.filter2D` with Prewitt kernels

10. **Watershed Segmentation**
    - Treats grayscale images as topographic surfaces and segments regions using water flow simulation.
    - Useful for separating overlapping objects.
    - **Implementation:** `cv2.watershed` with markers and distance transform

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Scikit-image (`skimage`)
