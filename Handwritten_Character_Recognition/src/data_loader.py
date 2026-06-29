import gzip
import os
import numpy as np


# --------------------------------------------------
# Dataset Path
# --------------------------------------------------
from src.config import DATA_DIR

# --------------------------------------------------
# Read Images
# --------------------------------------------------
def load_images(filepath):
    """
    Load image data from IDX file.
    """

    with gzip.open(filepath, "rb") as f:
        # Skip IDX header (16 bytes)
        _ = int.from_bytes(f.read(4), "big")
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")

        images = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        images = images.reshape(num_images, rows, cols)

    return images


# --------------------------------------------------
# Read Labels
# --------------------------------------------------
def load_labels(filepath):
    """
    Load labels from IDX file.
    """

    with gzip.open(filepath, "rb") as f:

        # Skip IDX header (8 bytes)
        _ = int.from_bytes(f.read(4), "big")
        num_labels = int.from_bytes(f.read(4), "big")

        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

    return labels


# --------------------------------------------------
# Load Entire Dataset
# --------------------------------------------------
def load_emnist():

    train_images = load_images(
        DATA_DIR / "emnist-letters-train-images-idx3-ubyte.gz"
    )

    train_labels = load_labels(
        DATA_DIR / "emnist-letters-train-labels-idx1-ubyte.gz"
    )

    test_images = load_images(
        DATA_DIR/ "emnist-letters-test-images-idx3-ubyte.gz"

    )

    test_labels = load_labels(
        DATA_DIR / "emnist-letters-test-labels-idx1-ubyte.gz"
    )

    return train_images, train_labels, test_images, test_labels


# --------------------------------------------------
# Test Loader
# --------------------------------------------------
from preprocess import preprocess_images, preprocess_labels


if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_emnist()

    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    y_train = preprocess_labels(y_train)
    y_test = preprocess_labels(y_test)

    print("=" * 50)
    print("EMNIST Dataset Loaded Successfully")
    print("=" * 50)

    print("Training Images:", X_train.shape)
    print("Training Labels:", y_train.shape)

    print("Testing Images :", X_test.shape)
    print("Testing Labels :", y_test.shape)

    print("\nPixel Range:")
    print(X_train.min(), X_train.max())

    print("\nUnique Labels:")
    print(np.unique(y_train))