"""Unit tests for Stage 1 data checks and utilities."""

import pandas as pd
import pytest

from src.mp055.data_checks import (
    check_duplicate_rows,
    check_leaking_column,
    check_sentinel_value,
    check_spelling_variants,
    check_stuck_sensor,
    check_target_typos,
    check_unit_mixup,
)
from src.mp055.io_utils import load_params, sha256_of


def test_params_loading():
    """Verify params.yaml can be safely loaded and contains required keys."""
    params = load_params("params.yaml")
    assert "data" in params
    assert "train_sha256" in params["data"]
    assert params["settings"]["random_seed"] == 42


def test_train_sha256_integrity():
    """Verify train.csv sha256 matches the recorded digest in params.yaml."""
    params = load_params("params.yaml")
    expected_sha = params["data"]["train_sha256"]
    computed_sha = sha256_of(params["data"]["raw_train_path"])
    assert computed_sha.lower() == expected_sha.lower()


def test_synthetic_data_checks():
    """Verify data checks logic on controlled synthetic DataFrames."""
    # Test duplicate detection
    df_dups = pd.DataFrame({"a": [1, 2, 2, 3]})
    res_dup = check_duplicate_rows(df_dups)
    assert res_dup["present"] is True
    assert res_dup["count"] == 1

    # Test sentinel detection
    df_sentinel = pd.DataFrame({"a": [10, 20, -1] * 10})
    res_sentinel = check_sentinel_value(df_sentinel)
    assert res_sentinel["present"] is True
    assert res_sentinel["value"] == -1
    assert res_sentinel["count"] == 10

    # Test target typos detection
    df_typos = pd.DataFrame({"kg_per_worker": [20.0] * 50 + [200.0, 300.0]})
    res_typos = check_target_typos(df_typos)
    assert res_typos["present"] is True
    assert res_typos["count"] == 2
