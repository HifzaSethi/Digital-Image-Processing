## Overview

This project demonstrates the application of **classical image segmentation techniques** using Python and the **scikit-image** library. The focus is on understanding and implementing **Active Contours (Snakes)** and **Level Set (Chan–Vese)** segmentation methods for object boundary detection.

## Techniques Implemented

### 1️⃣ Active Contour (Snake) Segmentation

- Converted input images to grayscale and applied Gaussian smoothing to reduce noise.
- Used Otsu thresholding and morphological operations to obtain initial object regions.
- Extracted contours and refined them using the **Active Contour (Snake) algorithm**.
- Applied snakes to multiple objects within a single image.

**Purpose:**
To accurately capture object boundaries by iteratively evolving contours based on image gradients and smoothness constraints.

### 2️⃣ Chan–Vese Level Set Segmentation

- Applied the **Chan–Vese level set method**, which does not rely on image gradients.
- Used a checkerboard initialization for automatic segmentation.
- Segmented objects based on regional intensity differences.
- Visualized segmentation results by overlaying contours on the original image.

**Purpose:**
To segment objects with weak or blurred edges where edge-based methods may fail.

## Libraries & Tools Used

- Python
- NumPy
- Matplotlib
- scikit-image

  - `active_contour`
  - `chan_vese`
  - `filters`, `morphology`, `measure`

## Learning Outcomes

Through this project, I learned:

- The practical differences between edge-based and region-based segmentation
- How preprocessing (smoothing, thresholding) affects segmentation quality
- The working principles of Active Contours and Level Set methods
- How to visualize and evaluate segmentation results effectively

## Academic Context

This work is part of **Digital Image Processing (DIP)** practice, focusing on **classical segmentation algorithms** before transitioning to deep-learning-based approaches.
