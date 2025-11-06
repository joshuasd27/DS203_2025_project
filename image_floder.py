import os
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from color_hist import extract_color_histogram_features

def process_image_folder_color_only(
    image_folder,
    save_path,
    extensions=(".jpg", ".png", ".jpeg", ".bmp")
):
    """
    Process all images in a folder using color histogram feature extraction.

    Parameters
    ----------
    image_folder : str or Path
        Path to the folder containing images.
    save_path : str or Path
        Path to save the final DataFrame (.pkl file).
    extensions : tuple
        Image file extensions to include.

    Returns
    -------
    pd.DataFrame
        Combined features for all images.
    """

    image_folder = Path(image_folder)
    all_images = [p for p in image_folder.iterdir() if p.suffix.lower() in extensions]

    if not all_images:
        raise ValueError(f"No valid images found in {image_folder}")

    combined_features = []

    print(f"📂 Found {len(all_images)} images. Starting color histogram extraction...")

    for img_path in tqdm(all_images, desc="Extracting color histograms", unit="image"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path.name}")
            continue

        try:
            fd = extract_color_histogram_features(img)
            fd["image_name"] = img_path.name
            combined_features.append(fd)
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    if not combined_features:
        raise RuntimeError("⚠️ No valid images processed!")

    final_df = pd.concat(combined_features, ignore_index=True)
    final_df.to_pickle(save_path)

    print("\n✅ Color histogram extraction complete!")
    print("📊 Final DataFrame shape:", final_df.shape)
    print(f"💾 Saved to: {save_path}")

    return final_df


if __name__ == "__main__":
    image_dir = r"E:\Sem_3\DS 203\E7\resized_images"
    save_path = r"E:\Sem_3\DS 203\E7\feature_outputs\gridwise_color_features.pkl"

    df_all = process_image_folder_color_only(image_dir, save_path)
