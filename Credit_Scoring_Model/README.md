# 💳 Credit Scoring Model

A Machine Learning project developed for the **CodeAlpha Machine Learning Internship**.

This application predicts whether a customer is likely to experience serious financial distress (Bad Credit) or maintain good creditworthiness (Good Credit) based on their financial information.

---

## 📌 Project Overview

Credit scoring is an important application of Machine Learning in the banking and finance industry. This project analyzes customer financial data and predicts credit risk using different classification algorithms.

The project compares multiple machine learning models and selects the best-performing model for deployment in a Streamlit web application.

---

## 🎯 Objectives

- Predict customer creditworthiness.
- Compare multiple classification models.
- Evaluate model performance.
- Deploy the best model using Streamlit.

---

## 📂 Dataset

**Dataset:** Give Me Some Credit

The dataset contains customer financial information including:

- Revolving Credit Utilization
- Age
- Debt Ratio
- Monthly Income
- Open Credit Lines
- Late Payment History
- Real Estate Loans
- Number of Dependents

Target Variable:

- **0 → Good Credit**
- **1 → Bad Credit**

The dataset used in this project is publicly available on Kaggle:

**Give Me Some Credit**

Download the dataset and place `cs-training.csv` inside:

data/raw/

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Streamlit
- Joblib

---

## 🤖 Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest ✅ (Selected)

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|---------:|----------:|--------:|----------:|---------:|
| Logistic Regression | 69.00% | 13.36% | 66.08% | 22.22% | 74.74% |
| Decision Tree | 86.65% | 25.41% | 51.30% | 33.99% | 80.43% |
| Random Forest | **87.83%** | **28.09%** | **52.30%** | **36.55%** | **83.10%** |

Random Forest was selected as the final model because it achieved the best overall balance between Accuracy, Recall, F1-Score, and ROC-AUC.

---

## 📁 Project Structure

```text
Credit-Scoring-Model/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│   ├── plots/
│   └── model_comparison.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── visualization.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone <repository-link>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

---

## 📈 Features

- Data Cleaning
- Missing Value Handling
- Duplicate Removal
- SMOTE for Class Balancing
- Multiple Model Comparison
- Model Evaluation
- Streamlit Dashboard
- Credit Risk Prediction

---

## 🔮 Future Improvements

- Hyperparameter Optimization
- XGBoost and LightGBM Models
- Explainable AI using SHAP
- Cloud Deployment
- REST API Integration

---

## 👨‍💻 Author

Developed as part of the **CodeAlpha Machine Learning Internship**.