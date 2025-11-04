import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from math import ceil

# Parameters for LBP
radius = 3
n_points = 8 * radius
method = 'uniform'  # options: 'default', 'ror', 'uniform', 'var'

# Paths
image_dir = "E:\\Sem_3\\DS 203\\E7\\resized_images"  # folder containing your images
save_path = "lbp_features.pkl"

# Create list of image files
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Empty list to store results
feature_list = []

# Process each image
for img_file in tqdm(image_files, desc="Processing Images"):
    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # Resize if larger than 800x600
    

    # Divide into 64 equal grids (8x8)
    grid_rows, grid_cols = 8, 8
    grid_h, grid_w = ceil(img.shape[0] / grid_rows), ceil(img.shape[1] / grid_cols)

    img_features = []
    for i in range(0, img.shape[0], grid_h):
        for j in range(0, img.shape[1], grid_w):
            grid = img[i:i + grid_h, j:j + grid_w]

            # Compute LBP
            lbp = local_binary_pattern(grid, n_points, radius, method)

            # Compute histogram of LBP
            hist, _ = np.histogram(lbp.ravel(),
                                   bins=np.arange(0, n_points + 3),
                                   range=(0, n_points + 2))

            # Normalize histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)

            # Append grid features
            img_features.extend(hist)
        print(f"Processed grid row {i // grid_h + 1}/{grid_rows} for image {img_file}")

    # Save feature vector for the image
    feature_entry = {"image": img_file}
    feature_entry.update({f"feature_{k}": v for k, v in enumerate(img_features)})
    feature_list.append(feature_entry)

# Convert to DataFrame
df = pd.DataFrame(feature_list)
print("✅ LBP Feature Extraction Complete! DataFrame shape:", df.shape)

# Save to pickle for fast reload
df.to_pickle(save_path)
print(f"💾 Saved to {save_path}")
