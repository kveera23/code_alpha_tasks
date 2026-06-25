import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_dataset

# ----------------------------------------
# Create Output Folder
# ----------------------------------------
os.makedirs("outputs", exist_ok=True)

# ========================================
# 1. Emotion Distribution
# ========================================

df = load_dataset()

plt.figure(figsize=(10, 6))

sns.countplot(
    x="emotion",
    data=df,
    order=df["emotion"].value_counts().index
)

plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/emotion_distribution.png",
    dpi=300
)

plt.close()

print("Emotion Distribution Saved")


# ========================================
# 2. MFCC Sample
# ========================================

X = np.load("data/processed/X_features.npy")

plt.figure(figsize=(12, 6))

plt.imshow(
    X[0],
    aspect="auto",
    origin="lower",
    cmap="viridis"
)

plt.colorbar()

plt.title("MFCC Feature Sample")

plt.xlabel("Time")

plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    "outputs/mfcc_sample.png",
    dpi=300
)

plt.close()

print("MFCC Sample Saved")


# ========================================
# 3. Confusion Matrix
# ========================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model #type:ignore
from tensorflow.keras.utils import to_categorical #type:ignore

X = np.load("data/processed/X_features.npy")
y = np.load("data/processed/y_labels.npy")

# Normalize
X = np.array([
    (sample - np.mean(sample)) / (np.std(sample) + 1e-8)
    for sample in X
])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

_, X_test, _, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

X_test = X_test[..., np.newaxis]

model = load_model("models/best_model.keras")

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(10, 8))

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion Matrix Saved")


# ========================================
# 4. Training Curves
# ========================================

history_path = "outputs/history.pkl"

if os.path.exists(history_path):

    with open(history_path, "rb") as f:
        history = pickle.load(f)

    # Accuracy
    plt.figure(figsize=(10, 5))

    plt.plot(
        history["accuracy"],
        label="Training Accuracy",
        linewidth=2
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2
    )

    plt.title("Training Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "outputs/accuracy_curve.png",
        dpi=300
    )

    plt.close()

    # Loss

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "outputs/loss_curve.png",
        dpi=300
    )

    plt.close()

    print("Training Curves Saved")

else:

    print("history.pkl not found. Training curves skipped.")

print("\nAll Visualizations Saved Successfully!")

