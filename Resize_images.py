import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.color import rgb2gray

def process_image(image_path):
    """
    Open image, resize if >800x600, divide into 8x8 grids,
    compute HOG features for each grid.
    Returns: list of feature vectors for all 64 grids.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return None

    h, w = img.shape[:2]

    # Resize if larger than 800x600
    if w > 800 or h > 600:
        img = cv2.resize(img, (800, 600))
        print(f"Resized {os.path.basename(image_path)} from ({w},{h}) to (800,600)")
    else:
        print(f"Kept original size for {os.path.basename(image_path)}: ({w},{h})")
    return img
'''
    gray = rgb2gray(img)

    num_rows, num_cols = 8, 8
    grid_h, grid_w = gray.shape[0] // num_rows, gray.shape[1] // num_cols

    hog_features_all = []
    for i in range(num_rows):
        for j in range(num_cols):
            y1, y2 = i * grid_h, (i + 1) * grid_h
            x1, x2 = j * grid_w, (j + 1) * grid_w
            patch = gray[y1:y2, x1:x2]

            features = hog(
                patch,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                transform_sqrt=True,
                feature_vector=True
            )

            hog_features_all.append(features)

    return hog_features_all'''


# ---------- MAIN SCRIPT ----------
if __name__ == "__main__":
    image_folder = "E:\Sem_3\DS 203\E7\DS203-2025-S1-E7-project-images\images"        # Folder containing images
    output_excel = "all_image_features.xlsx"
    output_csv = "all_image_features.csv"   # Optional, can disable if not needed
    count = 0
    all_features = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            count += 1
            image_path = os.path.join(image_folder, filename)
            img = process_image(image_path)
            cv2.imwrite(f"E:\\Sem_3\\DS 203\\E7\\resized_images\\image_{count}.jpg", img)
            '''
            if features is not None:
                for grid_idx, fvec in enumerate(features, start=1):
                    all_features.append({
                        "Image": filename,
                        "Grid_Index": grid_idx,
                        **{f"Feature_{k+1}": v for k, v in enumerate(fvec)}
                    })'''
            
'''
    # Combine all into one DataFrame
    df = pd.DataFrame(all_features)

    # Save to Excel and CSV
    df.to_excel(output_excel, index=False)
    df.to_csv(output_csv, index=False)

    print(f"\n✅ All image HOG features saved to:")
    print(f"   → {output_excel}")
    print(f"   → {output_csv}")'''
