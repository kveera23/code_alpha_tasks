from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_data():
    # Fetch Heart Disease dataset
    heart_disease = fetch_ucirepo(id=45)

    # Features and target
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine features and target
    df = pd.concat([X, y], axis=1)

    return df

if __name__ == "__main__":
    df = load_data()

    # Save locally
    df.to_csv("data/processed/heart_raw.csv", index=False)

    print("Dataset saved successfully!")
    print("Shape:", df.shape)