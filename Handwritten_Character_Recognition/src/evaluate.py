import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from data_loader import load_emnist
from preprocess import preprocess_images, preprocess_labels

from src.config import (
    MODEL_PATH,
    HISTORY_PATH,
    PLOT_DIR,
    REPORT_DIR
)

PLOT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Create Output Directories
# ==========================================================

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


# ==========================================================
# Load Model
# ==========================================================

print("=" * 60)
print("Loading Trained Model...")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)


# ==========================================================
# Load Dataset
# ==========================================================

_, _, X_test, y_test = load_emnist()

X_test = preprocess_images(X_test)
y_test = preprocess_labels(y_test)


# ==========================================================
# Predictions
# ==========================================================

print("\nPredicting...")

predictions = model.predict(X_test, verbose=1)

y_pred = np.argmax(predictions, axis=1)


# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ==========================================================
# Classification Report
# ==========================================================

letters = [chr(i) for i in range(65, 91)]

report = classification_report(
    y_test,
    y_pred,
    target_names=letters
)

print("\nClassification Report:\n")
print(report)

with open(
    "outputs/reports/classification_report.txt",
    "w"
) as f:

    f.write(report)


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=letters
)

fig, ax = plt.subplots(figsize=(14, 14))

disp.plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=90,
    colorbar=False
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================================
# Accuracy & Loss Curves
# ==========================================================

history = np.load(
    HISTORY_PATH,
    allow_pickle=True
).item()


plt.figure(figsize=(8,5))

plt.plot(
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "accuracy_curve.png",
    dpi=300
)

plt.close()



plt.figure(figsize=(8,5))

plt.plot(
    history["loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "loss_curve.png",
    dpi=300
)

plt.close()


# ==========================================================
# Prediction Examples
# ==========================================================

indices = np.random.choice(
    len(X_test),
    12,
    replace=False
)

plt.figure(figsize=(12,8))

for i, idx in enumerate(indices):

    plt.subplot(3,4,i+1)

    plt.imshow(
        X_test[idx].squeeze(),
        cmap="gray"
    )

    true_letter = chr(y_test[idx] + 65)
    pred_letter = chr(y_pred[idx] + 65)

    color = "green"

    if true_letter != pred_letter:
        color = "red"

    plt.title(
        f"T:{true_letter} P:{pred_letter}",
        color=color,
        fontsize=10
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "prediction_examples.png",
    dpi=300
)

plt.close()


print("\nEvaluation Completed Successfully!")

print("\nFiles Saved:")

print("✔ classification_report.txt")

print("✔ confusion_matrix.png")

print("✔ accuracy_curve.png")

print("✔ loss_curve.png")

print("✔ prediction_examples.png")