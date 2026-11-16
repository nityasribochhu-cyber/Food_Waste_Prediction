
import pandas as pd
from pathlib import Path


def load_donation_centers(path="data/donation_centers.csv"):
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(file_path)


def recommend_donation_centers(food_category, predicted_waste):
    centers = load_donation_centers()

    # Check required columns
    required_columns = ["center_name", "accepted_categories"]

    for column in required_columns:
        if column not in centers.columns:
            raise ValueError(
                f"Missing column in donation_centers.csv: {column}"
            )

    # Match food category
    matching_centers = centers[
        centers["accepted_categories"]
        .fillna("")
        .str.contains(food_category, case=False, regex=False)
    ].copy()

    # Check donation quantity if capacity column exists
    if "capacity_kg" in matching_centers.columns:
        matching_centers = matching_centers[
            pd.to_numeric(
                matching_centers["capacity_kg"],
                errors="coerce"
            ).fillna(0) >= predicted_waste
        ]

    return matching_centers
