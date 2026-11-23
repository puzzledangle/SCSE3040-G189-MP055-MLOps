"""Data cleaning pipeline for MP055 (Tea Leaf Plucking Indore).

Resolves anomalies identified during Stage 1 data checks:
1. Duplicate records removed.
2. Target leak column (kg_weighed_at_factory) dropped.
3. Target typos (values > 100 typed with an extra zero) corrected.
4. Sentinel value (-1) in rain_mm_week imputed with domain median.
5. Inconsistent flush spelling variants mapped to canonical categories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.mp055.io_utils import load_csv, load_params


def clean_flush_variants(series: pd.Series) -> pd.Series:
    """Normalize inconsistent flush category names to canonical levels.

    Args:
        series: Raw categorical Series of flush picking seasons.

    Returns:
        pd.Series: Cleaned Series with values in {'first', 'second', 'monsoon'}.
    """
    def _map_val(val: object) -> str:
        s = str(val).strip().lower().replace(".", "")
        if s.startswith("fir"):
            return "first"
        if s.startswith("sec"):
            return "second"
        if s.startswith("mon"):
            return "monsoon"
        return s

    return series.apply(_map_val)


def clean_dataset(
    df: pd.DataFrame,
    is_train: bool = True,
    impute_rain_val: Optional[float] = None,
) -> Tuple[pd.DataFrame, float]:
    """Execute full cleaning routine on raw dataframe.

    Args:
        df: Raw input DataFrame.
        is_train: True if processing training data, False for later/holdout.
        impute_rain_val: Pre-computed median rainfall value for test/later data.

    Returns:
        Tuple[pd.DataFrame, float]: Cleaned DataFrame and the rainfall imputation value.
    """
    cleaned = df.copy()

    # 1. Deduplication (only applicable when training/evaluating historical data)
    if is_train:
        cleaned = cleaned.drop_duplicates()

    # 2. Remove target-leaking column if present
    if "kg_weighed_at_factory" in cleaned.columns:
        cleaned = cleaned.drop(columns=["kg_weighed_at_factory"])

    # 3. Correct typographical errors in the target (extra zero typed)
    if is_train and "kg_per_worker" in cleaned.columns:
        typos_mask = cleaned["kg_per_worker"] > 100.0
        cleaned.loc[typos_mask, "kg_per_worker"] = (
            cleaned.loc[typos_mask, "kg_per_worker"] / 10.0
        )

    # 4. Handle sentinel values (-1) in rain_mm_week
    if "rain_mm_week" in cleaned.columns:
        cleaned["rain_mm_week"] = cleaned["rain_mm_week"].astype(float)
        sentinel_mask = cleaned["rain_mm_week"] == -1.0
        cleaned.loc[sentinel_mask, "rain_mm_week"] = np.nan

        if impute_rain_val is None:
            impute_rain_val = float(cleaned["rain_mm_week"].median())

        cleaned["rain_mm_week"] = cleaned["rain_mm_week"].fillna(impute_rain_val)
    else:
        impute_rain_val = 0.0

    # 5. Standardize spelling variants in flush
    if "flush" in cleaned.columns:
        cleaned["flush"] = clean_flush_variants(cleaned["flush"])

    return cleaned, impute_rain_val


def run_cleaning() -> pd.DataFrame:
    """Run cleaning routine on train.csv and save to cleaned_train_path."""
    params = load_params("params.yaml")
    train_path = params["data"]["raw_train_path"]
    out_path = Path(params["data"]["cleaned_train_path"])

    print(f"Loading raw training data from {train_path}...")
    raw_df = load_csv(train_path)
    print(f"Raw shape: {raw_df.shape}")

    cleaned_df, rain_median = clean_dataset(raw_df, is_train=True)
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Imputed rainfall median: {rain_median:.1f} mm")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(out_path, index=False)
    print(f"Cleaned dataset saved to: {out_path}")

    return cleaned_df


if __name__ == "__main__":
    run_cleaning()
