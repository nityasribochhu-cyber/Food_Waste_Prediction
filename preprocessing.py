
import pandas as pd

def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    return df


def clean_data(df):

    # 1. Remove duplicate rows
    df = df.drop_duplicates()

    # 2. Convert date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 3. Clean categorical values
    df["staff_experience"] = (
        df["staff_experience"].astype("string").str.strip().str.title()
    )

    df["waste_category"] = (
        df["waste_category"].astype("string").str.strip().str.title()
    )

    # 4. Handle missing numerical values
    numerical_cols = df.select_dtypes(include="number").columns

    for col in numerical_cols:
        if col != "food_waste_kg":
            df[col] = df[col].fillna(df[col].median())

    # 5. Handle missing categorical values
    categorical_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in categorical_cols:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 6. Remove rows where target is missing
    df = df.dropna(subset=["food_waste_kg"])

    return df


if __name__ == "__main__":

    df = load_data()

    print("Original Dataset Shape:", df.shape)

    cleaned_df = clean_data(df)

    print("\nCleaned Dataset Shape:", cleaned_df.shape)

    print("\nMissing Values After Cleaning:")
    print(cleaned_df.isnull().sum())

    cleaned_df.to_csv("data/cleaned_train.csv", index=False)

    print("\nCleaned dataset saved successfully!")
