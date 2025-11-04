import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog
from skimage.color import rgb2gray

def process_image(image_path):
    """Resize if >800x600, divide into 8x8 grids, compute HOG features."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return None

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
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                transform_sqrt=True,
                feature_vector=True
            )
            hog_features_all.append(features)
    return hog_features_all


if __name__ == "__main__":
    image_folder = "E:\\Sem_3\\DS 203\\E7\\resized_images"
    output_excel = "all_image_features.xlsx"
    output_csv = "all_image_features.csv"

    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    print(f"🔍 Found {len(image_files)} images to process...\n")

    all_rows = []
    # --- Step 1: Feature extraction with progress bar
    for filename in tqdm(image_files, desc="Extracting HOG features"):
        image_path = os.path.join(image_folder, filename)
        features = process_image(image_path)
        if features is None:
            continue
        for grid_idx, fvec in enumerate(features, start=1):
            row = {"Image": filename, "Grid_Index": grid_idx}
            for k, v in enumerate(fvec):
                row[f"F_{k+1}"] = v
            all_rows.append(row)

    # --- Step 2: Create DataFrame
    df = pd.DataFrame(all_rows)
    df.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)

    total_rows = len(df)

    df.to_pickle("hog_features_compressed.pkl.gz", compression="gzip")
    df.to_parquet("hog_features_compressed.parquet", compression="snappy")
    print(f"\n✅ Completed successfully!")
'''
    # --- Step 3: Write to Excel incrementally with progress
    chunk_size = 500  # number of rows per write batch
    num_chunks = (total_rows // chunk_size) + 1

    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("HOG_Features")
        writer.sheets["HOG_Features"] = worksheet

        # Write headers
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name)

        # Write data in chunks with progress bar
        start_row = 1
        for i in tqdm(range(num_chunks), desc="Writing to Excel"):
            start = i * chunk_size
            end = min(start + chunk_size, total_rows)
            chunk = df.iloc[start:end]
            for r, row in enumerate(chunk.itertuples(index=False, name=None)):
                for c, val in enumerate(row):
                    worksheet.write(start_row + r, c, val)
            start_row += len(chunk)

'''  


    #print(f"📘 Excel file: {output_excel}")
    #print(f"📄 CSV file:   {output_csv}")
