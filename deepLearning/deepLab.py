import torch
import torchvision.transforms as T
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load Pretrained DeepLabV3 Model
model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

# Read and preprocess image
image = cv2.imread(r"F:\python\AI\dip\Week5\deepLearning\indoor.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

input_tensor = transform(image).unsqueeze(0)


# Perform Segmentation
with torch.no_grad():
    output = model(input_tensor)['out']

segmentation_map = torch.argmax(output, dim=1).squeeze().cpu().numpy()


# Display Results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Original Indoor Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("DeepLab Semantic Segmentation")
plt.imshow(segmentation_map, cmap="jet")
plt.axis("off")

plt.show()
