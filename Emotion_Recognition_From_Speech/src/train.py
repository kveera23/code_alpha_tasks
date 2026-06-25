import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from joblib import dump
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore

# -----------------------------
# Load Features
# -----------------------------
X = np.load("data/processed/X_features.npy")
print("Feature Statistics")
print("-" * 30)

print("Minimum :", X.min())
print("Maximum :", X.max())
print("Mean    :", X.mean())
print("Std Dev :", X.std())
print("\nFirst Sample Shape:", X[0].shape)

print("\nFirst MFCC Matrix:")

print(X[0])

y = np.load("data/processed/y_labels.npy")

print("Features Loaded Successfully!\n")
print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)

# -----------------------------
# Normalize MFCC Features
# -----------------------------
# Normalize each sample independently
X = np.array([
    (sample - np.mean(sample)) / (np.std(sample) + 1e-8)
    for sample in X
])

print("MFCC Features Normalized Successfully!")

print("MFCC Features Normalized Successfully!")



# -----------------------------
# Encode Labels
# -----------------------------
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nEmotion Classes:")

for i, emotion in enumerate(label_encoder.classes_):
    print(f"{i} -> {emotion}")



dump(label_encoder, "models/label_encoder.pkl")

print("Label Encoder Saved!")

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)



# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Add channel dimension for CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("\nCNN Input Shape:", X_train.shape)

from model import build_cnn_model

model = build_cnn_model(
    input_shape=X_train.shape[1:],
    num_classes=y_train.shape[1]
)

model.summary()



early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)




# -----------------------------
# Train CNN Model
# -----------------------------
checkpoint = ModelCheckpoint(
    "models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=32,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],
    verbose=1
)



callbacks=[
    early_stopping,
    reduce_lr,
    checkpoint
]

os.makedirs("models", exist_ok=True)

model.save("models/cnn_emotion_model.keras")

print("\n✅ Model saved successfully!")
print("Location: models/cnn_emotion_model.keras")


import pickle
import os

os.makedirs("outputs", exist_ok=True)

with open("outputs/history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Training history saved successfully!")