**Image Processing: Noise Removal, Morphological Operations, and Object Extraction**

This assignment contains a Python project for processing PBM images using noise removal, morphological operations, skeletonization, and object extraction. The code demonstrates how to clean a noisy binary image, extract the main object, and draw its bounding box.
**Features**

**Load PBM Image**

Reads a PBM (Portable Bitmap) image and converts it to a binary image.

**Noise Removal**

Applies median filtering to remove salt-and-pepper noise.

**Morphological Operations**

Opening: Removes small white noise.

Closing: Closes small gaps and holes in the object.

**Skeletonization and Branch Removal**

Skeletonizes the main object.

Removes small branches and bridges to clean the skeleton.

**Normalization**

Normalizes pixel values for better visualization.

**Object Mask Extraction**

Thresholding and small object removal to extract the main object.

**Bounding Box Creation**

Determines object coordinates and draws a bounding box around the object.

Visualization

Displays all processing steps using matplotlib for easy comparison.
