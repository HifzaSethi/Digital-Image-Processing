# Assignment#6 DeepLearning Segmentation

## Overview

This project demonstrates various deep learning-based image segmentation techniques using PyTorch, torchvision, Segmentation Models PyTorch (SMP), U-Net, and YOLOv8. The assignment explores semantic, instance, and panoptic-style segmentation on different images, including indoor scenes, roads, and synthetic datasets.

## Segmentation Models Implemented

1. **DeepLabV3 (ResNet-50)**

   - Type: Semantic Segmentation
   - Framework: `torchvision.models.segmentation`
   - Input: Indoor image
   - Output: Segmentation map of the indoor scene
   - Notes: Uses pretrained weights for accurate pixel-wise classification.

2. **FCN (Fully Convolutional Network, ResNet-50)**

   - Type: Semantic Segmentation
   - Input: Generic RGB image
   - Output: Semantic segmentation using `tab20` colormap
   - Notes: Demonstrates semantic segmentation for multiple classes.

3. **Mask R-CNN**

   - Type: Instance Segmentation (Panoptic-style demo)
   - Input: Road image
   - Output: Segmentation of specific object classes (e.g., cars, bicycles)
   - Notes: Focused on “thing” classes with instance masks and transparency overlay.

4. **SegNet-style U-Net (SMP)**

   - Type: Semantic Segmentation
   - Input: Indoor image
   - Output: Segmentation map (floor, wall, chair, door, ceiling, background)
   - Notes: Encoder-decoder architecture using VGG16 encoder as in SegNet.

5. **Custom U-Net**

   - Type: Semantic Segmentation (binary mask)
   - Input: Custom image (e.g., water vs. land)
   - Output: Segmentation mask
   - Notes: Trained on synthetic mask data to demonstrate U-Net training and inference.

6. **YOLOv8 Segmentation**
   - Type: Instance / Panoptic-style Segmentation
   - Input: Image for object detection and segmentation
   - Output: Segmentation masks, bounding boxes, class labels, and confidence scores
   - Notes: Leverages ultralytics YOLOv8 pretrained segmentation models.

## Dependencies

- Python >= 3.8
- PyTorch
- torchvision
- segmentation-models-pytorch (SMP)
- ultralytics (YOLOv8)
- OpenCV
- matplotlib
- PIL (Pillow)
- NumPy

Install dependencies via pip:

```bash
pip install torch torchvision segmentation-models-pytorch ultralytics opencv-python matplotlib pillow numpy
```

## Usage

1. Update the image paths in each script to point to your dataset:

   ```python
   image_path = r"F:\python\AI\dip\Week5\deepLearning\your_image.jpg"
   ```

2. Run the Python script for the desired segmentation model:

   ```bash
   python deeplabv3_segmentation.py
   python fcn_segmentation.py
   python maskrcnn_segmentation.py
   python unet_smp_segmentation.py
   python custom_unet_training.py
   python yolov8_segmentation.py
   ```

3. Visualizations will display using `matplotlib` with original and segmented images side by side.

## Project Highlights

- Demonstrates end-to-end deep learning segmentation workflows.
- Covers semantic, instance, and panoptic-style segmentation.
- Implements both pretrained models and custom U-Net training.
- Uses visualizations for qualitative evaluation of segmentation results.
- Provides practical examples of indoor, road, and synthetic images.

## References

- [PyTorch Segmentation Models](https://pytorch.org/vision/stable/models.html)
- [Segmentation Models PyTorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
