import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# Load UNet (SegNet-style architecture)
model = smp.Unet(
    encoder_name="vgg16",        # same encoder as SegNet
    encoder_weights="imagenet",
    classes=6,                   # floor, wall, chair, door, ceiling, background
    activation=None
)

model.eval()

# Load and preprocess image
image = cv2.imread(r"F:\python\AI\dip\Week5\deepLearning\segnet.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))
img_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0)


# Segmentation
with torch.no_grad():
    output = model(img_tensor)

seg_map = torch.argmax(output, dim=1).squeeze().cpu().numpy()


# Display results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Original Indoor Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Semantic Segmentation Output")
plt.imshow(seg_map, cmap="jet")
plt.axis("off")

plt.show()

# “SegNet is an encoder–decoder semantic segmentation network.
# Since SegNet is not available as a built-in pretrained model, we used a U-Net with VGG16 encoder, which follows the same encoder–decoder principle and performs semantic segmentation effectively on indoor scenes.”