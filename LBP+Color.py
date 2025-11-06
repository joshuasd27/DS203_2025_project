import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import ceil

# ------------------------------
# PATHS
# ------------------------------
image_dir = "E:\\Sem_3\\DS 203\\E7\\resized_images"
lbp_pickle_path = r"E:\Sem_3\DS 203\E7\feature_outputs\gridwise_lbp_features.pkl"
save_path = "E:\\Sem_3\\DS 203\\E7\\feature_outputs\\combined_lbp_color.pkl"

# ------------------------------
# LOAD EXISTING LBP FEATURES
# ------------------------------
print("📂 Loading LBP features...")
df_lbp = pd.read_pickle(lbp_pickle_path)
print("✅ LBP shape:", df_lbp.shape)

# ------------------------------
# PARAMETERS
# ------------------------------
grid_rows, grid_cols = 8, 8
color_bins = 32  # 32 bins per channel is a good balance (3*32 = 96 features total)

# ------------------------------
# COLOR HISTOGRAM EXTRACTOR
# ------------------------------
def extract_color_histograms(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grid_h, grid_w = ceil(img.shape[0] / grid_rows), ceil(img.shape[1] / grid_cols)
    grid_color_features = []
    grid_id = 0

    for i in range(0, img.shape[0], grid_h):
        for j in range(0, img.shape[1], grid_w):
            grid = img[i:i + grid_h, j:j + grid_w]
            if grid.size == 0:
                continue

            # Compute RGB histograms
            hist_r = cv2.calcHist([grid], [0], None, [color_bins], [0, 256])
            hist_g = cv2.calcHist([grid], [1], None, [color_bins], [0, 256])
            hist_b = cv2.calcHist([grid], [2], None, [color_bins], [0, 256])

            # Normalize
            hist_r = hist_r / (hist_r.sum() + 1e-6)
            hist_g = hist_g / (hist_g.sum() + 1e-6)
            hist_b = hist_b / (hist_b.sum() + 1e-6)

            color_features = np.concatenate([hist_r, hist_g, hist_b]).flatten()
            grid_color_features.append({
                "grid_id": grid_id,
                **{f"color_{k}": v for k, v in enumerate(color_features)}
            })
            grid_id += 1

    return pd.DataFrame(grid_color_features)


# ------------------------------
# BUILD COMBINED FEATURES
# ------------------------------
combined_entries = []
image_names = df_lbp["image_name"].unique()

for img_file in tqdm(image_names, desc="Combining LBP + Color Hist"):
    img_path = os.path.join(image_dir, img_file)
    df_img_lbp = df_lbp[df_lbp["image_name"] == img_file].reset_index(drop=True)
    df_color = extract_color_histograms(img_path)

    if df_color is None:
        continue

    # Merge on grid_id
    df_combined = pd.merge(df_img_lbp, df_color, on="grid_id", how="left")
    combined_entries.append(df_combined)

# ------------------------------
# CONCATENATE & SAVE
# ------------------------------
df_combined_all = pd.concat(combined_entries, ignore_index=True)
df_combined_all.to_pickle(save_path)

print("\n✅ Combination Complete!")
print("Final shape:", df_combined_all.shape)
print(f"💾 Saved combined LBP + Color features to {save_path}")
