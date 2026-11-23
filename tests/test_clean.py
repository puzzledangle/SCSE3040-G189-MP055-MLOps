"""Tests for data cleaning pipeline (Stage 2)."""

import numpy as np
import pandas as pd
import pytest

from src.mp055.clean import clean_dataset, clean_flush_variants


def test_clean_flush_variants():
    """Verify categorical spelling variants are mapped to canonical levels."""
    raw_flush = pd.Series(["first", "fir.", "FIRST", "monsoon", "MONSOON", "mon.", "second", "sec.", "SECOND"])
    cleaned = clean_flush_variants(raw_flush)
    assert set(cleaned.unique()) == {"first", "second", "monsoon"}


def test_clean_dataset_drops_leak_and_duplicates():
    """Verify leaking column and duplicate rows are purged."""
    raw = pd.DataFrame({
        "record_id": ["R1", "R1", "R2"],
        "worker_exp_yr": [5, 5, 10],
        "rain_mm_week": [100.0, 100.0, 150.0],
        "slope_deg": [10, 10, 20],
        "temp_c": [22.0, 22.0, 25.0],
        "bush_age_yr": [30, 30, 40],
        "hours_worked": [7.0, 7.0, 8.0],
        "flush": ["first", "first", "monsoon"],
        "section": ["Upper Garden", "Upper Garden", "River Side"],
        "kg_weighed_at_factory": [35.0, 35.0, 42.0],
        "kg_per_worker": [35.2, 35.2, 41.8],
    })
    cleaned, _ = clean_dataset(raw, is_train=True)
    assert len(cleaned) == 2  # duplicate dropped
    assert "kg_weighed_at_factory" not in cleaned.columns


def test_clean_dataset_corrects_typos_and_sentinels():
    """Verify target typos (>100) are divided by 10 and sentinels (-1) are imputed."""
    raw = pd.DataFrame({
        "record_id": ["R1", "R2", "R3"],
        "worker_exp_yr": [5, 10, 15],
        "rain_mm_week": [-1.0, 120.0, 140.0],
        "slope_deg": [10, 20, 15],
        "temp_c": [22.0, 25.0, 24.0],
        "bush_age_yr": [30, 40, 35],
        "hours_worked": [7.0, 8.0, 7.5],
        "flush": ["fir.", "sec.", "mon."],
        "section": ["Upper Garden", "River Side", "Lower Garden"],
        "kg_per_worker": [350.0, 28.0, 42.0],  # 350.0 is a typo
    })
    cleaned, rain_median = clean_dataset(raw, is_train=True)
    # Check typo correction
    assert cleaned.loc[0, "kg_per_worker"] == 35.0
    # Check sentinel imputation
    assert cleaned.loc[0, "rain_mm_week"] == rain_median
    assert rain_median == 130.0  # median of [120, 140]
