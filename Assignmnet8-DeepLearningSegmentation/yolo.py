from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# -----------------------------------
# Load YOLOv8 Segmentation Model
# -----------------------------------
# You can choose "yolov8s-seg.pt" (small, fast) or any other seg model
model = YOLO("yolov8s-seg.pt")  


# Load and prepare input image

img_path = r"F:\python\AI\dip\Week5\deepLearning\panoptic.jpg"
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Run segmentation

results = model(image_rgb)


# Show original image

plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Show segmentation results

# YOLOv8 returns masks, boxes, labels, scores
seg_mask = results[0].masks
boxes = results[0].boxes
class_ids = results[0].boxes.cls
scores = results[0].boxes.conf

# Plot segmented output
plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)

# Draw segmentation masks
if seg_mask is not None:
    for i, mask in enumerate(seg_mask.data):
        # show mask with transparency
        plt.imshow(mask.cpu().numpy(), alpha=0.4)

        # get label name
        cls_id = int(class_ids[i].cpu().numpy())
        score = float(scores[i].cpu().numpy())

        # text annotation
        label = model.names[cls_id]
        plt.text(
            boxes.data[i][0], boxes.data[i][1] - 5,
            f"{label} {score:.2f}",
            color="yellow", fontsize=10, backgroundcolor="black"
        )

plt.title("YOLOv8 Segmentation Output")
plt.axis("off")
plt.show()
