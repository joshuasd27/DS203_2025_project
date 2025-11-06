import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# ------------------------------
# PATHS
# ------------------------------
save_dir = "E:\\Sem_3\\DS 203\\E7\\feature_outputs"

hog_path = os.path.join(save_dir, "gridwise_hog_features.pkl")
lbp_path = os.path.join(save_dir, "gridwise_lbp_features.pkl")
combined_path = os.path.join(save_dir, "gridwise_combined_hog_lbp_features.pkl")

# ------------------------------
# LOAD DATA
# ------------------------------
print("📂 Loading HOG and LBP pickles...")
df_hog = pd.read_pickle(hog_path)
df_lbp = pd.read_pickle(lbp_path)

print("✅ HOG shape:", df_hog.shape)
print("✅ LBP shape:", df_lbp.shape)

# ------------------------------
# ALIGN METADATA
# ------------------------------
# We'll merge on image_name + grid_id (they should match)
merge_keys = ["image_name", "grid_id"]

# Sanity check: ensure both have the same identifiers
if not set(merge_keys).issubset(df_hog.columns) or not set(merge_keys).issubset(df_lbp.columns):
    raise KeyError("Both DataFrames must contain 'image_name' and 'grid_id' columns for merging.")

# ------------------------------
# SELECT FEATURE COLUMNS
# ------------------------------
hog_features = [c for c in df_hog.columns if "feature_" in c]
lbp_features = [c for c in df_lbp.columns if "feature_" in c]

# ------------------------------
# MERGE FEATURES
# ------------------------------
df_combined = pd.merge(
    df_hog[merge_keys + hog_features],
    df_lbp[merge_keys + lbp_features],
    on=merge_keys,
    how='inner'
)

# ------------------------------
# NORMALIZE (z-score)
# ------------------------------
scaler = StandardScaler()
feature_cols = [c for c in df_combined.columns if "feature_" in c]
df_combined[feature_cols] = scaler.fit_transform(df_combined[feature_cols])

# ------------------------------
# SAVE COMBINED DATA
# ------------------------------
df_combined.to_pickle(combined_path)
print(f"\n✅ Combined HOG+LBP DataFrame created!")
print("Shape:", df_combined.shape)
print(f"💾 Saved to {combined_path}\n")

# Optional: quick check
print(df_combined.head(3))
