import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore


# ------------------------------------
# Create Output Folder
# ------------------------------------
os.makedirs("outputs", exist_ok=True)


# ------------------------------------
# Load Features
# ------------------------------------
X = np.load("data/processed/X_features.npy")
y = np.load("data/processed/y_labels.npy")


# ------------------------------------
# Normalize Features
# ------------------------------------
X = np.array([
    (sample - np.mean(sample)) / (np.std(sample) + 1e-8)
    for sample in X
])


# ------------------------------------
# Encode Labels
# ------------------------------------
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


# ------------------------------------
# Train-Test Split
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# Add channel dimension
X_test = X_test[..., np.newaxis]

# One-hot labels
y_test_cat = to_categorical(y_test)


# ------------------------------------
# Load Model
# ------------------------------------
model = load_model("models/best_model.keras")


# ------------------------------------
# Prediction
# ------------------------------------
predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)


# ------------------------------------
# Evaluation Metrics
# ------------------------------------
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


print("=" * 50)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("=" * 50)


# ------------------------------------
# Save Metrics
# ------------------------------------
with open("outputs/metrics.txt", "w") as f:
    f.write("Emotion Recognition From Speech\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")


# ------------------------------------
# Classification Report
# ------------------------------------
report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)

print("\nClassification Report\n")
print(report)

with open("outputs/classification_report.txt", "w") as f:
    f.write(report)


# ------------------------------------
# Confusion Matrix
# ------------------------------------

print("\nFiles Saved Successfully!")
print("--------------------------------")
print("outputs/metrics.txt")
print("outputs/classification_report.txt")
