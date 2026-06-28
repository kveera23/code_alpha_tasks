import joblib
import pandas as pd

model = joblib.load("models/credit_scoring_model.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/model_name.txt", "r") as f:
    model_name = f.read().strip()


def predict_credit_score(input_data):

    input_df = pd.DataFrame([input_data])

    # Scale only for Logistic Regression
    if model_name == "Logistic Regression":
        input_data = scaler.transform(input_df)
    else:
        input_data = input_df

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    result = "Good Credit" if prediction == 0 else "Bad Credit"

    confidence = max(probability)

    return result, confidence