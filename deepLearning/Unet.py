import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# -----------------------------
# Simple U-Net
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.mid = DoubleConv(128, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c2 = DoubleConv(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        m = self.mid(self.p2(d2))
        u2 = self.u2(m)
        u2 = self.c2(torch.cat([u2, d2], 1))
        u1 = self.u1(u2)
        u1 = self.c1(torch.cat([u1, d1], 1))
        return self.out(u1)



# Load Image
img_path = r"F:\python\AI\dip\Week5\deepLearning\Unet.jpg"
image = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

img_tensor = transform(image)

# -----------------------------
# Create Synthetic Mask (Water vs Land)
# -----------------------------
img_np = np.array(image.resize((256,256)))
blue_channel = img_np[:, :, 2]

mask = (blue_channel > 120).astype(np.int64)  # water = 1, land = 0
mask = torch.from_numpy(mask)

# -----------------------------
# Prepare Data
# -----------------------------
x = img_tensor.unsqueeze(0)
y = mask.unsqueeze(0)

# -----------------------------
# Train U-Net
# -----------------------------
model = UNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(30):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/30  Loss: {loss.item():.4f}")

# -----------------------------
# Inference
# -----------------------------
model.eval()
with torch.no_grad():
    pred = model(x).argmax(1).squeeze().numpy()

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mask, cmap="gray")
plt.title("Synthetic Ground Truth Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(pred, cmap="gray")
plt.title("U-Net Prediction")
plt.axis("off")

plt.show()
