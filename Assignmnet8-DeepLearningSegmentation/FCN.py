import torch
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Step 1: Load pretrained FCN model
model = models.segmentation.fcn_resnet50(pretrained=True)
model.eval()  # set model to evaluation mode

# Step 2: Load and preprocess image
image_path = r"F:\python\AI\dip\Week5\deepLearning\FCN.jpg"  
image = Image.open(image_path).convert("RGB")

transform = T.Compose([
    T.Resize((520, 520)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

input_tensor = transform(image).unsqueeze(0)  # add batch dimension


# Step 3: Perform semantic segmentation
with torch.no_grad():
    output = model(input_tensor)["out"]

# Get predicted class for each pixel
segmentation = output.argmax(1).squeeze().cpu().numpy()


# Step 4: Display results
plt.figure(figsize=(12, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Segmented image
plt.subplot(1, 2, 2)
plt.imshow(segmentation, cmap="tab20")
plt.title("Semantic Segmentation (FCN)")
plt.axis("off")

plt.tight_layout()
plt.show()
