import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import ceil
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler

def Gaussian_smooth(img, sigma=1.0, strength=1.0):
    """
    Apply Gaussian sharpening to the image using unsharp masking.
    
    Parameters:
        img       : Input image (BGR or RGB)
        sigma     : Standard deviation for Gaussian blur
        strength  : Sharpening intensity (0.5–2.0 typical)
    """
    # Gaussian blur the image
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    
    # Unsharp mask: original + strength * (original - blurred)
    sharpened = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    
    return sharpened
# ------------------------------
# FEATURE EXTRACTION FUNCTION
# ------------------------------
def extract_features(mode="hog", img_in = None, image_dir="", save_dir="", Gaus_blur = True):
    """
    mode: "hog" or "lbp"
    """
    # ------------------------------
    # PARAMETERS
    # ------------------------------
    hog_params = dict(
        orientations=9,                 # standard HOG orientation bins
        pixels_per_cell=(8, 8),         # smaller cells = more spatial detail
        cells_per_block=(2, 2),         # normalization region
        block_norm='L2-Hys',
        transform_sqrt=True
    )

    radius = 2
    n_points = 8 * radius
    lbp_method = 'uniform'


    gridwise_features = []
    #flattened_features = []
    scaler = StandardScaler()
    
    if img_in is None:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_files = [img_in]
    for img_index, img_file in enumerate(tqdm(image_files, desc=f"Extracting {mode.upper()} features")):
        if img_in is None:
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
        else:
            img = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)

        
        if img is None:
            continue

        img = Gaussian_smooth(img) if Gaus_blur else img
        grid_rows, grid_cols = 8, 8
        grid_h, grid_w = ceil(img.shape[0] / grid_rows), ceil(img.shape[1] / grid_cols)

        img_all_features = []
        grid_id = 0

        for i in range(0, img.shape[0], grid_h):
            for j in range(0, img.shape[1], grid_w):
                grid = img[i:i + grid_h, j:j + grid_w]
                if grid.size == 0:
                    continue

                if mode == "hog":
                    # --- HOG ---
                    features = hog(grid, **hog_params, feature_vector=True)

                elif mode == "lbp":
                    # --- LBP ---
                    lbp = local_binary_pattern(grid, n_points, radius, lbp_method)
                    features, _ = np.histogram( lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2) )
                    #features, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
                    features = features.astype("float")
                    features /= (features.sum() + 1e-6)

                # --- Store grid features ---
                entry = {
                    "image_id": f"Image_{img_index + 1}",
                    "image_name": img_file,
                    "grid_id": grid_id
                }
                entry.update({f"feature_{k}": v for k, v in enumerate(features)})
                gridwise_features.append(entry)

                # --- Add to image-level list ---
                img_all_features.extend(features)
                grid_id += 1

        # store one long feature vector per image
        #flat_entry = {"image_id": f"Image_{img_index + 1}", "image_name": img_file}
        #flat_entry.update({f"feature_{k}": v for k, v in enumerate(img_all_features)})
        #flattened_features.append(flat_entry)

    # ------------------------------
    # DataFrames
    # ------------------------------
    df_grid = pd.DataFrame(gridwise_features)
    #df_flat = pd.DataFrame(flattened_features)

    # --- Global normalization ---
    feature_cols = [c for c in df_grid.columns if "feature_" in c]
    df_grid[feature_cols] = scaler.fit_transform(df_grid[feature_cols])

    #feature_cols_flat = [c for c in df_flat.columns if "feature_" in c]
    #df_flat[feature_cols_flat] = scaler.fit_transform(df_flat[feature_cols_flat])

    # --- Save outputs ---
    #print(type(save_dir))
    df_grid.to_pickle(os.path.join(save_dir, f"256_gridwise_{mode}_features.pkl"))
    #df_flat.to_pickle(os.path.join(save_dir, f"flattened_{mode}_features.pkl"))

    print(f"\n✅ {mode.upper()} Extraction Complete!")
    print("Grid-wise shape:", df_grid.shape)
    #print("Flattened shape:", df_flat.shape)
    #print(f"💾 Saved to {save_dir}\n")
    return df_grid

# ------------------------------
# RUN FOR HOG AND LBP
# ------------------------------
#extract_features("hog")



if __name__ == "__main__":
    

    # ------------------------------
    # PATHS
    # ------------------------------
    image_dir = "E:\\Sem_3\\DS 203\\E7\\resized_images"
    save_dir = "E:\\Sem_3\\DS 203\\E7\\feature_outputs_fine"
    os.makedirs(save_dir, exist_ok=True)

    #image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    extract_features("lbp", image_dir = image_dir, save_dir=save_dir)

