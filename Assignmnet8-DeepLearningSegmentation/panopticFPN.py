# Due to limited Windows support for panoptic models, instance segmentation using Mask R-CNN was used to demonstrate panoptic principles by separating thing classes such as cars and bicycles, while treating road as a stuff class.

# Due to limited Windows support for panoptic models, instance segmentation using Mask R-CNN
# was used to demonstrate panoptic principles by separating thing classes such as cars and
# bicycles, while treating road as a stuff class.

import torch
import torchvision
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt


# Load Mask R-CNN Model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess image
image = cv2.imread(r"F:\python\AI\dip\Week5\deepLearning\Road.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transform = T.Compose([
    T.ToTensor()
])

input_tensor = transform(image)


# Instance Segmentation
with torch.no_grad():
    output = model([input_tensor])[0]


# Visualization
plt.figure(figsize=(14, 6))

# 1. Original Image 
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# 2. Segmented Image
plt.subplot(1, 2, 2)
plt.imshow(image)

for i in range(len(output["masks"])):
    score = output["scores"][i]
    label = output["labels"][i]

    # COCO labels: 3 = bicycle, 4 = car
    if score > 0.6 and label in [3, 4]:
        mask = output["masks"][i, 0].cpu().numpy()
        plt.imshow(mask, alpha=0.5)

plt.title("Panoptic-Style Segmentation (Car & Bicycle)")
plt.axis("off")

plt.show()
