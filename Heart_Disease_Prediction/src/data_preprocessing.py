import pandas as pd

# Load dataset
df = pd.read_csv("data/processed/heart_raw.csv")

# Convert target to binary
df['num'] = (df['num'] > 0).astype(int)

# Fill missing values with median
df['ca'] = df['ca'].fillna(df['ca'].median())
df['thal'] = df['thal'].fillna(df['thal'].median())

print("Missing Values After Cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("data/processed/heart_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")