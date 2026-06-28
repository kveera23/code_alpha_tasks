# ==========================================
# Credit Scoring Model
# Data Preprocessing
# ==========================================

import os
import pandas as pd

# Create folders if they don't exist
os.makedirs("data/processed", exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/cs-training.csv")

# Remove unnecessary index column if present
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# ------------------------------------------
# Basic Information
# ------------------------------------------

print("=" * 50)
print("First 5 Rows")
print("=" * 50)
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# ------------------------------------------
# Missing Values
# ------------------------------------------

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# ------------------------------------------
# Remove Duplicate Rows
# ------------------------------------------

duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

df.drop_duplicates(inplace=True)

# ------------------------------------------
# Handle Missing Values
# ------------------------------------------

# Fill MonthlyIncome with median
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(
    df["MonthlyIncome"].median()
)

# Fill NumberOfDependents with median
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
    df["NumberOfDependents"].median()
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ------------------------------------------
# Separate Features and Target
# ------------------------------------------

X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

# ------------------------------------------
# Save Clean Dataset
# ------------------------------------------

clean_df = pd.concat([X, y], axis=1)

clean_df.to_csv(
    "data/processed/credit_data_clean.csv",
    index=False
)

print("\nClean dataset saved successfully!")