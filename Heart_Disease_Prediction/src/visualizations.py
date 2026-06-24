import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Load dataset
df = pd.read_csv("data/processed/heart_cleaned.csv")

X = df.drop("num", axis=1)
y = df["num"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -------------------------
# Confusion Matrix
# -------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")
plt.savefig("reports/images/confusion_matrix.png")
plt.close()

# -------------------------
# ROC Curve
# -------------------------

RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("ROC Curve")
plt.savefig("reports/images/roc_curve.png")
plt.close()


importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))

importance.plot(kind="bar")

plt.title("Feature Importance")
plt.tight_layout()

plt.savefig("reports/images/feature_importance.png")
plt.close()

print("Visualizations saved successfully!")