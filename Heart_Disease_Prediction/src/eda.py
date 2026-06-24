import pandas as pd

df = pd.read_csv("data/processed/heart_raw.csv")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())
print(df['num'].value_counts())
print(df.columns.tolist())