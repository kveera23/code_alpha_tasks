# ==========================================
# Credit Scoring Model
# Model Training and Evaluation
# ==========================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ------------------------------------------
# Create folders
# ------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# ------------------------------------------
# Load cleaned dataset
# ------------------------------------------

df = pd.read_csv("data/processed/credit_data_clean.csv")

# Features and target
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

# ------------------------------------------
# Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples :", X_train.shape)
print("Testing Samples  :", X_test.shape)

# ------------------------------------------
# Handle Class Imbalance using SMOTE
# ------------------------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE")
print("Training Samples :", X_train_smote.shape)

print("\nTarget Distribution:")
print(y_train_smote.value_counts())


# ------------------------------------------
# Feature Scaling
# ------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# ------------------------------------------
# Models
# ------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=10
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = []

best_model = None
best_model_name = ""
best_accuracy = 0

# ------------------------------------------
# Train and Evaluate
# ------------------------------------------

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Logistic Regression uses scaled SMOTE data
    if name == "Logistic Regression":

        model.fit(X_train_scaled, y_train_smote)

        predictions = model.predict(X_test_scaled)

        probabilities = model.predict_proba(X_test_scaled)[:, 1]

    # Decision Tree & Random Forest use SMOTE data without scaling
    else:

        model.fit(X_train_smote, y_train_smote)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ])

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Good", "Bad"]
    )

    disp.plot()

    plt.title(name)
    plt.savefig(f"outputs/plots/{name.replace(' ', '_')}_cm.png")
    plt.close()

    # Save Best Model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# ------------------------------------------
# Save Best Model
# ------------------------------------------

joblib.dump(best_model, "models/credit_scoring_model.pkl")

print("\nBest Model:", best_model_name)
print(f"Best Accuracy: {best_accuracy:.4f}")

# ------------------------------------------
# Results Table
# ------------------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]
)

print("\nModel Comparison")
print(results_df)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")

with open("models/model_name.txt", "w") as f:
    f.write(best_model_name)