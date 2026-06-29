import numpy as np

def preprocess_images(images):
    """
    Preprocess EMNIST images for CNN.
    """

    # Rotate 90 degrees counter-clockwise
    images = np.rot90(images, k=1, axes=(1, 2))

    # Flip horizontally
    images = np.fliplr(images)

    # Normalize
    images = images.astype(np.float32) / 255.0

    # Add channel dimension
    images = np.expand_dims(images, axis=-1)

    return images


def preprocess_labels(labels):
    """
    Convert labels from 1-26 to 0-25.
    """

    return labels - 1