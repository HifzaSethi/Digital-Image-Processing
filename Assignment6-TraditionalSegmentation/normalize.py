import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.segmentation import slic, mark_boundaries
from skimage import graph

# Step 1: Read the image
image = img_as_float(io.imread(r'F:\python\AI\dip\Week5\Traditional\normalize.jpg'))  

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
plt.show()

# Step 2: Apply SLIC superpixel segmentation first (required for N-cut graph)
labels1 = slic(image, compactness=30, n_segments=400, start_label=0)

# Step 3: Build RAG (Region Adjacency Graph)
g = graph.rag_mean_color(image, labels1, mode='similarity')

# Step 4: Apply normalized cut
labels2 = graph.cut_normalized(labels1, g)

# Step 5: Show segmented image with boundaries
out = mark_boundaries(image, labels2)

plt.figure(figsize=(8, 6))
plt.imshow(out)
plt.title("Normalized Cut Segmentation")
plt.axis('off')
plt.show()
