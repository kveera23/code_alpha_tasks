import os
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_emnist
from preprocess import preprocess_images, preprocess_labels

# ==========================================================
# Create Output Directory
# ==========================================================

os.makedirs("outputs/plots", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

X_train, y_train, _, _ = load_emnist()

X_train = preprocess_images(X_train)
y_train = preprocess_labels(y_train)

letters = [chr(i) for i in range(65, 91)]

# ==========================================================
# 1. Sample Images
# ==========================================================

plt.figure(figsize=(12, 8))

for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(X_train[i].squeeze(), cmap="gray")
    plt.title(letters[y_train[i]])
    plt.axis("off")

plt.suptitle("Sample EMNIST Letter Images", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    "outputs/plots/sample_images.png",
    dpi=300
)

plt.close()

# ==========================================================
# 2. Class Distribution
# ==========================================================

counts = np.bincount(y_train, minlength=26)

plt.figure(figsize=(14, 6))

plt.bar(
    letters,
    counts
)

plt.title("Class Distribution of EMNIST Letters")

plt.xlabel("Letters")

plt.ylabel("Number of Samples")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "outputs/plots/class_distribution.png",
    dpi=300
)

plt.close()

print("=" * 50)
print("Visualizations Generated Successfully!")
print("=" * 50)

print("Saved:")
print("✔ sample_images.png")
print("✔ class_distribution.png")