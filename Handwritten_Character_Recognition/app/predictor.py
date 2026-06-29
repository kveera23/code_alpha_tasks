from PIL import Image

from src.predict import predict_character


def predict_uploaded_image(uploaded_file):
    """
    Predict handwritten character from uploaded image.
    """

    image = Image.open(uploaded_file)

    predicted_letter, confidence = predict_character(image)

    return image, predicted_letter, confidence