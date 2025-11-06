import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1️⃣ Load Pickle File
# ------------------------------
# 🔸 Change this path to whichever file you want to inspect:
pkl_path = r"E:\Sem_3\DS 203\E7\feature_outputs\gridwise_hog_features.pkl"

# Load the DataFrame
df = pd.read_pickle(pkl_path)
print("✅ Data Loaded Successfully!")
print("📂 File Path:", pkl_path)

# ------------------------------
# 2️⃣ Basic Information
# ------------------------------
print("\n--- BASIC INFO ---")
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print("\nData Types:")
print(df.dtypes.value_counts())

# ------------------------------
# 3️⃣ Sample Data
# ------------------------------
print("\n--- DATA HEAD (First 5 Rows) ---")
print(df.head())

print("\n--- RANDOM SAMPLE (5 Rows) ---")
print(df.sample(5, random_state=42))

# ------------------------------
# 4️⃣ Statistical Summary
# ------------------------------
print(df.columns)
numeric_cols = [c for c in df.columns if "color_" in c or "feature_" in c] #change this condition based on your feature column naming
print("\n--- STATISTICAL SUMMARY (Numeric Columns) ---")
print(df[numeric_cols].describe())

# ------------------------------
# 5️⃣ Check for Missing or Infinite Values
# ------------------------------
print("\n--- MISSING VALUE CHECK ---")
missing = df.isnull().sum().sum()
infinite = ((df[numeric_cols] == float("inf")) | (df[numeric_cols] == -float("inf"))).sum().sum()
print(f"Missing values: {missing}, Infinite values: {infinite}")

# ------------------------------
# 6️⃣ Optional: Correlation Heatmap (for first 30 features)
# ------------------------------
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numeric_cols[:30]].corr(), cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap (All Features)")
    plt.show()

# ------------------------------
# 7️⃣ Optional: Quick Histogram of a Random Feature
# ------------------------------
sample_feature = numeric_cols[0] if numeric_cols else None
if sample_feature:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[sample_feature], bins=30, kde=True)
    plt.title(f"Histogram of {sample_feature}")
    plt.show()
