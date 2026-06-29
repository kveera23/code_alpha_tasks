import cv2
import numpy as np
import tensorflow as tf

from PIL import Image

from src.config import MODEL_PATH

# ==========================================================
# Load Model
# ==========================================================

model = tf.keras.models.load_model(MODEL_PATH)

letters = [chr(i) for i in range(65, 91)]


# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_uploaded_image(image):
    """
    Preprocess uploaded handwritten image to match EMNIST format.
    """

    # ==========================================
    # PIL -> NumPy
    # ==========================================

    image = image.convert("L")
    image = np.array(image)

    # ==========================================
    # Invert if background is white
    # ==========================================

    if np.mean(image) > 127:
        image = 255 - image

    # ==========================================
    # Binary Threshold
    # ==========================================

    _, image = cv2.threshold(
        image,
        30,
        255,
        cv2.THRESH_BINARY
    )

    # ==========================================
    # Find Character Bounding Box
    # ==========================================

    coords = cv2.findNonZero(image)

    if coords is None:

        image = np.zeros((28, 28), dtype=np.uint8)

    else:

        x, y, w, h = cv2.boundingRect(coords)

        image = image[y:y+h, x:x+w]

        # ======================================
        # Resize while keeping aspect ratio
        # ======================================

        target = 20

        height, width = image.shape

        if height > width:

            new_height = target
            new_width = int(width * target / height)

        else:

            new_width = target
            new_height = int(height * target / width)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        # ======================================
        # Pad to 28x28
        # ======================================

        canvas = np.zeros((28, 28), dtype=np.uint8)

        y_offset = (28 - new_height) // 2
        x_offset = (28 - new_width) // 2

        canvas[
            y_offset:y_offset+new_height,
            x_offset:x_offset+new_width
        ] = image

        image = canvas

    # ==========================================
    # Normalize
    # ==========================================

    image = image.astype(np.float32) / 255.0

    # ==========================================
    # Add Dimensions
    # ==========================================

    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image




# ==========================================================
# Prediction Function
# ==========================================================

def predict_character(image):
    """
    Predict handwritten character.

    Parameters
    ----------
    image : PIL.Image

    Returns
    -------
    tuple
        (Predicted Letter, Confidence)
    """

    processed_image = preprocess_uploaded_image(image)

    cv2.imwrite(
        "outputs/preprocessed_image.png",
        (processed_image[0] * 255).astype(np.uint8)
    )

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    predicted_class = np.argmax(prediction)

    confidence = float(prediction[0][predicted_class])

    predicted_letter = letters[predicted_class]

    return predicted_letter, confidence

if __name__ == "__main__":

    image = Image.open("test.png")

    letter, confidence = predict_character(image)

    print("=" * 50)
    print("Prediction")
    print("=" * 50)

    print("Character :", letter)
    print(f"Confidence: {confidence*100:.2f}%")