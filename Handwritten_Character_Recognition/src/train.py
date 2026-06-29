import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Input
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from data_loader import load_emnist
from preprocess import preprocess_images, preprocess_labels

from src.config import (
    MODEL_DIR,
    MODEL_PATH,
    OUTPUT_DIR,
    HISTORY_PATH
)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading EMNIST Dataset...")
print("=" * 60)

X_train, y_train, X_test, y_test = load_emnist()

X_train = preprocess_images(X_train)
X_test = preprocess_images(X_test)

y_train = preprocess_labels(y_train)
y_test = preprocess_labels(y_test)

print("Training Images :", X_train.shape)
print("Testing Images  :", X_test.shape)

print("Training Labels :", y_train.shape)
print("Testing Labels  :", y_test.shape)

# ==========================================================
# Build CNN Model
# ==========================================================

model = Sequential([

    Input(shape=(28, 28, 1)),

    Conv2D(32, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),

    Flatten(),

    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(26, activation="softmax")
])

# ==========================================================
# Compile
# ==========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================================
# Callbacks
# ==========================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

checkpoint = ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)

# ==========================================================
# Train
# ==========================================================

history = model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=20,

    batch_size=64,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],

    verbose=1
)

# ==========================================================
# Evaluate
# ==========================================================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("=" * 60)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")
print("=" * 60)

# ==========================================================
# Save Training History
# ==========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(
    HISTORY_PATH,
    history.history
)

print("\nTraining history saved.")
print("Best model saved in models/cnn_model.keras")