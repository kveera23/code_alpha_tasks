import joblib
import pandas as pd

# Load model
model = joblib.load("models/heart_disease_model.pkl")

def predict_heart_disease(input_data):
    """
    Predict heart disease and return prediction + probability
    """

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    return prediction, probability