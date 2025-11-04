import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --------------------------
# 1. Load Pickle file
# --------------------------
df = pd.read_pickle("E:\\Sem_3\\DS 203\\E7\\Code\\hog_features_compressed.pkl\\hog_features_compressed.pkl")  # change path if needed
print("✅ Data loaded successfully! Shape:", df.shape)

# Remove non-numeric columns (like filenames or grid indices)
numeric_df = df.select_dtypes(include=['number'])

# --------------------------
# 2. Standardize the data
# --------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_df)

# --------------------------
# 3. Perform PCA (2 components for 2D visualization)
# --------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# Add PCA columns back to dataframe for plotting
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

print(f"Explained variance by 2 components: {pca.explained_variance_ratio_.sum():.2%}")

# --------------------------
# 4. PCA Scatter Plot
# --------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='PCA1', y='PCA2',
    data=df,
    alpha=0.7, s=40
)
plt.title("PCA Scatter Plot of HOG Features (2D)")
plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.grid(True)
plt.show()

# --------------------------
# 5. Optional: 3D PCA Plot
# --------------------------
from mpl_toolkits.mplot3d import Axes3D

pca3 = PCA(n_components=3)
pca3_result = pca3.fit_transform(scaled_features)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca3_result[:, 0], pca3_result[:, 1], pca3_result[:, 2],
           alpha=0.6, s=40)
ax.set_title("3D PCA Scatter Plot of HOG Features")
ax.set_xlabel(f"PCA1 ({pca3.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PCA2 ({pca3.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PCA3 ({pca3.explained_variance_ratio_[2]*100:.1f}%)")
plt.show()
