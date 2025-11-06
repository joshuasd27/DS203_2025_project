import cv2
import numpy as np
import pandas as pd
from math import ceil
from skimage.feature import local_binary_pattern
from tqdm import tqdm
from pathlib import Path

def extract_lbp_color_features(
    img,
    grid_rows=8,
    grid_cols=8,
    lbp_radius=2,
    lbp_points=8,
    lbp_method='uniform',
    color_bins=32
):
    """
    Extract combined LBP + Color Histogram features from an OpenCV image object.

    Parameters:
    -----------
    img : np.ndarray
        Input image (BGR or RGB).
    grid_rows, grid_cols : int
        Number of grid splits.
    lbp_radius, lbp_points : int
        Parameters for LBP.
    lbp_method : str
        Method for LBP ('default', 'uniform', etc.).
    color_bins : int
        Number of bins per color channel for color histograms.

    Returns:
    --------
    pd.DataFrame : DataFrame with combined LBP + color histogram features per grid.
    """

    # Ensure image is valid
    if img is None or img.size == 0:
        raise ValueError("Invalid image provided!")

    # Convert to grayscale for LBP and RGB for color hist
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define grid sizes
    grid_h, grid_w = ceil(img.shape[0] / grid_rows), ceil(img.shape[1] / grid_cols)

    all_grids = []
    grid_id = 0

    for i in range(0, img.shape[0], grid_h):
        for j in range(0, img.shape[1], grid_w):
            grid_gray = img_gray[i:i + grid_h, j:j + grid_w]
            grid_rgb = img_rgb[i:i + grid_h, j:j + grid_w]

            if grid_gray.size == 0:
                continue

            # ----- LBP -----
            lbp = local_binary_pattern(grid_gray, lbp_points, lbp_radius, lbp_method)
            lbp_hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, lbp_points + 3),
                range=(0, lbp_points + 2)
            )
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-6)

            # ----- COLOR HISTOGRAM -----
            hist_r = cv2.calcHist([grid_rgb], [0], None, [color_bins], [0, 256])
            hist_g = cv2.calcHist([grid_rgb], [1], None, [color_bins], [0, 256])
            hist_b = cv2.calcHist([grid_rgb], [2], None, [color_bins], [0, 256])

            hist_r /= (hist_r.sum() + 1e-6)
            hist_g /= (hist_g.sum() + 1e-6)
            hist_b /= (hist_b.sum() + 1e-6)

            color_features = np.concatenate([hist_r, hist_g, hist_b]).flatten()

            # Combine features
            feature_dict = {
                "grid_id": grid_id,
                **{f"lbp_{k}": v for k, v in enumerate(lbp_hist)},
                **{f"color_{k}": v for k, v in enumerate(color_features)}
            }
            all_grids.append(feature_dict)
            grid_id += 1

    return pd.DataFrame(all_grids)

def process_image_folder(
    image_folder,
    save_path,
    extensions=(".jpg", ".png", ".jpeg", ".bmp"),
):
    """
    Process all images in a folder to extract LBP + Color Histogram features.

    Parameters
    ----------
    image_folder : str or Path
        Path to the folder containing images.
    save_path : str or Path
        Path to save the combined feature DataFrame as .pkl
    extensions : tuple
        Valid image extensions to include.

    Returns
    -------
    pd.DataFrame : Combined feature DataFrame for all images.
    """

    image_folder = Path(image_folder)
    all_images = [p for p in image_folder.iterdir() if p.suffix.lower() in extensions]

    if not all_images:
        raise ValueError(f"❌ No images found in {image_folder}")

    combined_features = []

    print(f"📂 Found {len(all_images)} images. Starting feature extraction...")

    for img_path in tqdm(all_images, desc="Extracting features", unit="image"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path.name}")
            continue

        try:
            fd = extract_lbp_color_features(img)
            fd["image_name"] = img_path.name
            combined_features.append(fd)
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    if not combined_features:
        raise RuntimeError("⚠️ No valid images processed!")

    final_df = pd.concat(combined_features, ignore_index=True)
    final_df.to_pickle(save_path)

    print("\n✅ Feature extraction complete!")
    print("📊 Final DataFrame shape:", final_df.shape)
    print(f"💾 Saved to: {save_path}")

    return final_df


if __name__ == "__main__":
    image_dir = r"E:\Sem_3\DS 203\E7\resized_images"
    save_path = r"E:\Sem_3\DS 203\E7\feature_outputs\new_gridwise_lbp_color_features.pkl"

    df_all = process_image_folder(image_dir, save_path)
