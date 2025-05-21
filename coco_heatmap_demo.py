import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from simple_bottom_up_pose import ScaleAwareHeatmapGenerator
import os

# Ensure coco_data folder exists
os.makedirs("coco_data", exist_ok=True)

# Place the extracted 'val2017' images folder and 'annotations' folder inside 'coco_data'.
coco_ann_path = "coco_data/annotations_trainval2017/annotations/person_keypoints_val2017.json"
coco_img_dir = "coco_data/val2017/val2017/"

# Load COCO annotations
with open(coco_ann_path, "r") as f:
    coco = json.load(f)

# Get the first person annotation with all keypoints visible
for ann in coco["annotations"]:
    if ann["num_keypoints"] == 17:
        keypoints = np.array(ann["keypoints"]).reshape(17, 3)
        image_id = ann["image_id"]
        break

# Get image file name
img_info = next(img for img in coco["images"] if img["id"] == image_id)
img_path = coco_img_dir + img_info["file_name"]

# Assign sigmas (use your scale-aware values or a fixed list)
sigmas = [2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4]
ct_sigma = 2

# Prepare joints array: (x, y, v, sigma)
joints = np.zeros((1, 17, 4))
joints[0, :, 0:2] = keypoints[:, 0:2]
joints[0, :, 2] = (keypoints[:, 2] > 0).astype(np.float32)
joints[0, :, 3] = sigmas

# Debug: print keypoints and joint visibility
print("Keypoints:", keypoints)
print("Joints (vis):", joints[0, :, 2])

output_res = 64  # Define this before using it

# Scale keypoints from image coordinates to heatmap coordinates
img = Image.open(img_path)
img_width, img_height = img.size
scale_x = output_res / img_width
scale_y = output_res / img_height
joints[0, :, 0] *= scale_x
joints[0, :, 1] *= scale_y

# Generate heatmaps
generator = ScaleAwareHeatmapGenerator(output_res, 17)
heatmaps, _ = generator(joints, sigmas, ct_sigma)

# Load the image
img = Image.open(img_path)

# Overlay keypoints and skeleton on the original image
# COCO skeleton definition: pairs of keypoint indices (0-based)
coco_skeleton = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]
fig_overlay, ax_overlay = plt.subplots(figsize=(6, 6))
ax_overlay.imshow(img)
ax_overlay.set_title("Original Image with COCO Keypoints & Skeleton")
ax_overlay.axis('off')
# Draw skeleton lines
for idx1, idx2 in coco_skeleton:
    x1, y1, v1 = keypoints[idx1]
    x2, y2, v2 = keypoints[idx2]
    if v1 > 0 and v2 > 0:
        ax_overlay.plot([x1, x2], [y1, y2], color='cyan', linewidth=2, zorder=2)
# Draw keypoints
for i, (x, y, v) in enumerate(keypoints):
    if v > 0:
        ax_overlay.scatter(x, y, color='lime', s=60, edgecolors='black', linewidths=1.5, zorder=3)
        ax_overlay.text(x, y, str(i+1), color='red', fontsize=10, zorder=4)
plt.tight_layout()
plt.show()

# Visualize original image and all 17 keypoint heatmaps in a single figure
fig, axes = plt.subplots(4, 5, figsize=(18, 12))

# Show original image in the first subplot
axes[0, 0].imshow(img)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Show heatmaps for each keypoint with debug prints
for i in range(17):
    row = (i + 1) // 5
    col = (i + 1) % 5
    # Debug: print min/max values for each heatmap
    print(f"Heatmap {i+1}: min={heatmaps[i].min()}, max={heatmaps[i].max()}")
    axes[row, col].imshow(heatmaps[i], cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    axes[row, col].set_title(f'Keypoint {i+1}')
    axes[row, col].axis('off')

# Hide any unused subplots
for j in range(18, 20):
    row = j // 5
    col = j % 5
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
