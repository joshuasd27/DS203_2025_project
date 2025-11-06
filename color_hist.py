import cv2
import numpy as np
import pandas as pd
from math import ceil

def extract_color_histogram_features(
    img,
    grid_rows=8,
    grid_cols=8,
    color_bins=32
):
    """
    Extract color histogram features (R, G, B) for each grid in the image.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or RGB).
    grid_rows, grid_cols : int
        Number of grid splits (default 8x8 = 64 grids).
    color_bins : int
        Number of bins per color channel.

    Returns
    -------
    pd.DataFrame
        DataFrame containing grid-wise color histogram features.
    """

    # Validate input
    if img is None or img.size == 0:
        raise ValueError("Invalid image provided!")

    # Convert to RGB for consistent color order
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define grid size
    grid_h, grid_w = ceil(img.shape[0] / grid_rows), ceil(img.shape[1] / grid_cols)

    all_grids = []
    grid_id = 0

    for i in range(0, img.shape[0], grid_h):
        for j in range(0, img.shape[1], grid_w):
            grid_rgb = img_rgb[i:i + grid_h, j:j + grid_w]

            if grid_rgb.size == 0:
                continue

            # Compute normalized color histograms for each channel
            hist_r = cv2.calcHist([grid_rgb], [0], None, [color_bins], [0, 256])
            hist_g = cv2.calcHist([grid_rgb], [1], None, [color_bins], [0, 256])
            hist_b = cv2.calcHist([grid_rgb], [2], None, [color_bins], [0, 256])

            hist_r /= (hist_r.sum() + 1e-6)
            hist_g /= (hist_g.sum() + 1e-6)
            hist_b /= (hist_b.sum() + 1e-6)

            # Concatenate all channels
            color_features = np.concatenate([hist_r, hist_g, hist_b]).flatten()

            # Store grid feature
            feature_dict = {
                "grid_id": grid_id,
                **{f"color_{k}": v for k, v in enumerate(color_features)}
            }
            all_grids.append(feature_dict)
            grid_id += 1

    return pd.DataFrame(all_grids)
