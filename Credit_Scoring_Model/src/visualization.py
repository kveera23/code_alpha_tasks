# ==========================================
# Credit Scoring Model
# Visualizations
# ==========================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay

# Create folder
os.makedirs("outputs/plots", exist_ok=True)

# Load data
df = pd.read_csv("data/processed/credit_data_clean.csv")

# Load model
model = joblib.load("models/credit_scoring_model.pkl")

# -----------------------------
# 1. Class Distribution
# -----------------------------

plt.figure(figsize=(6,4))
df["SeriousDlqin2yrs"].value_counts().plot(kind="bar")

plt.title("Class Distribution")
plt.xlabel("Credit Status")
plt.ylabel("Count")

plt.savefig("outputs/plots/class_distribution.png")
plt.close()

# -----------------------------
# 2. Correlation Heatmap
# -----------------------------

plt.figure(figsize=(10,8))

plt.imshow(df.corr(), cmap="coolwarm")

plt.colorbar()

plt.xticks(
    range(len(df.columns)),
    df.columns,
    rotation=90
)

plt.yticks(
    range(len(df.columns)),
    df.columns
)

plt.tight_layout()

plt.savefig("outputs/plots/correlation_heatmap.png")
plt.close()

# -----------------------------
# 3. Feature Importance
# -----------------------------

if hasattr(model, "feature_importances_"):

    importance = pd.Series(
        model.feature_importances_,
        index=df.drop("SeriousDlqin2yrs", axis=1).columns
    )

    importance.sort_values().plot(
        kind="barh",
        figsize=(8,6)
    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/feature_importance.png"
    )

    plt.close()

import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/plots/correlation_heatmap.png")

print("Visualizations created successfully!")

