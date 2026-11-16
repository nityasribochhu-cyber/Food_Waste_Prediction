

import pandas as pd

df = pd.read_csv("data/train.csv")

print("DATASET LOADED SUCCESSFULLY")
print("Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nColumn Data Types:")
print(df.dtypes)

print("\nFood Waste Statistics:")
print(df["food_waste_kg"].describe())
