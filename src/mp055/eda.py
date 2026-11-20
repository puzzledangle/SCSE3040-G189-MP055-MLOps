"""Read-only Exploratory Data Analysis (EDA) helper for MP055.

Inspects shapes, data types, distributions, null counts, and column
mismatches without modifying any source files.
"""

from __future__ import annotations

import pandas as pd
from src.mp055.io_utils import load_csv, load_params


def run_eda() -> None:
    """Print high-level descriptive summaries of train and holdout datasets."""
    params = load_params("params.yaml")
    train_path = params["data"]["raw_train_path"]
    holdout_path = params["data"]["raw_holdout_path"]

    df_train = load_csv(train_path)
    df_holdout = load_csv(holdout_path)

    print("=" * 70)
    print("MP055 EXPLORATORY DATA ANALYSIS (READ-ONLY)")
    print("=" * 70)

    print(f"\n1. DATASET SHAPES:")
    print(f"   train.csv            : {df_train.shape[0]} rows x {df_train.shape[1]} columns")
    print(f"   holdout_features.csv : {df_holdout.shape[0]} rows x {df_holdout.shape[1]} columns")

    print(f"\n2. DATA TYPES IN TRAIN:")
    print(df_train.dtypes)

    print(f"\n3. SAMPLE RECORDS (HEAD):")
    print(df_train.head(5))

    print(f"\n4. NUMERICAL SUMMARY (DESCRIBE):")
    print(df_train.describe().T)

    print(f"\n5. NULL COUNTS PER COLUMN:")
    nulls = df_train.isnull().sum()
    print(nulls[nulls > 0] if (nulls > 0).any() else "   No explicit NaN / null values found.")

    print(f"\n6. COLUMNS IN TRAIN BUT MISSING FROM HOLDOUT:")
    missing = [c for c in df_train.columns if c not in df_holdout.columns]
    print(f"   {missing}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_eda()
