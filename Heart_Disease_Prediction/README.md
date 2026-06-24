# ❤️ Heart Disease Prediction Using Machine Learning

## Project Overview

This project predicts the likelihood of heart disease using Machine Learning techniques. The model is trained on the UCI Heart Disease dataset and provides risk predictions based on patient clinical information.

The application includes data preprocessing, model training, evaluation, visualization, and deployment through a Streamlit web application.

---

## Objectives

- Predict the presence of heart disease using patient medical data.
- Compare multiple machine learning algorithms.
- Evaluate model performance using standard classification metrics.
- Deploy the best-performing model using Streamlit.

---

## Dataset

**Source:** UCI Machine Learning Repository

**Dataset:** Heart Disease Dataset

The dataset contains 303 patient records with 13 medical attributes and one target variable.

### Features

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate Achieved
- Exercise Induced Angina
- Oldpeak
- Slope of Peak Exercise ST Segment
- Number of Major Vessels
- Thalassemia Type

### Target

- 0 = No Heart Disease
- 1 = Heart Disease

---

## Machine Learning Algorithms Used

1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)

---

## Model Performance

| Model | Accuracy |
|---------|----------|
| Logistic Regression | 86.89% |
| Random Forest | 88.52% |
| SVM | 85.25% |

### Best Model

**Random Forest Classifier**

Performance Metrics:

- Accuracy: 88.52%
- Precision: 83.87%
- Recall: 92.86%
- F1 Score: 88.14%
- ROC-AUC: 95.18%

---

## Project Structure

```text
Heart_Disease_Prediction/

├── app/
│   └── streamlit_app.py

├── data/
│   └── processed/

├── models/
│   └── heart_disease_model.pkl

├── reports/
│   └── images/

├── src/
│   ├── load_dataset.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── model_comparison.py
│   ├── save_model.py
│   ├── visualizations.py
│   ├── predict.py
│   └── __init__.py

├── requirements.txt

└── README.md
```

---

## Visualizations

The project generates:

- Confusion Matrix
- ROC Curve
- Feature Importance Plot

These visualizations help evaluate model performance and identify important medical features influencing predictions.

---

## Streamlit Application

The web application allows users to:

- Enter patient medical details
- Predict heart disease risk
- View risk probability
- Receive Low, Moderate, or High Risk assessment

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app/streamlit_app.py
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- UCI ML Repository

---

## Author

K. Veera

B.Tech Student | Aspiring AI & Machine Learning Engineer

GitHub: https://github.com/kveera23

Developed as part of a Machine Learning Internship Project.