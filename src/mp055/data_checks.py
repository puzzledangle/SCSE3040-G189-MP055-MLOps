"""Data check implementations for MP055 (Tea Leaf Plucking Indore).

Investigates the seven data quality issues defined in the Student Handbook (Section 7).
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd


def check_leaking_column(
    df_train: pd.DataFrame,
    df_holdout: Optional[pd.DataFrame] = None,
    target_col: str = "kg_per_worker",
) -> dict[str, Any]:
    """Detect columns populated post-event that leak target information.

    Meaning:
        A leaking column is filled in only AFTER the event occurs, so it secretly
        contains the answer. A model trained using it appears deceptively accurate
        in training but fails in production. It is typically missing from holdout
        feature sets (holdout_features.csv).

    Expected Return Format (per findings.yaml):
        {
            "present": bool,
            "column": Optional[str]  # name of the leaking column, or None if absent
        }

    Hint:
        Compare `set(df_train.columns) - set(df_holdout.columns)` while excluding
        the target column itself. Then verify high correlation or small residual
        difference with the target.

    Gotcha:
        The target column itself is naturally missing from holdout_features.csv,
        so make sure to filter out `target_col` when identifying leaking candidates.
    """
    if df_holdout is not None:
        missing_in_holdout = [
            c for c in df_train.columns if c not in df_holdout.columns and c != target_col
        ]
        for col in missing_in_holdout:
            if col in df_train.columns:
                return {"present": True, "column": col}

    # Fallback heuristic: check for columns nearly identical to the target
    for col in df_train.columns:
        if col == target_col or not np.issubdtype(df_train[col].dtype, np.number):
            continue
        corr = df_train[col].corr(df_train[target_col])
        if corr > 0.95:
            return {"present": True, "column": col}

    return {"present": False, "column": None}


def check_unit_mixup(
    df: pd.DataFrame,
    group_col: str = "section",
) -> dict[str, Any]:
    """Detect if one operational section recorded a feature in a different unit.

    Meaning:
        One recording office or section logged values in a different unit (e.g.
        metres vs km, grams vs kg, months vs years). One group's values are roughly
        10x, 12x, 60x, 100x, or 1000x larger or smaller.

    Expected Return Format (per findings.yaml):
        {
            "present": bool,
            "column": Optional[str],
            "group_value": Optional[str],
            "factor": Optional[float]  # wrong value = right value x factor
        }

    Hint:
        Use `df.groupby(group_col)[col].describe()` or `.agg(['mean', 'median', 'max'])`
        to compare scale across groups.

    Gotcha:
        The `factor` must be an exact physical unit conversion constant (such as 1000,
        12, 60), not an empirical statistical ratio. If no group exhibits an order-of-magnitude
        conversion factor shift, the issue is absent.
    """
    standard_factors = [10.0, 12.0, 60.0, 100.0, 1000.0]
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        if col in ["kg_per_worker", "kg_weighed_at_factory"]:
            continue
        grp_means = df.groupby(group_col)[col].mean()
        if len(grp_means) < 2:
            continue
        overall_median = grp_means.median()
        if overall_median == 0:
            continue
        for grp_val, val in grp_means.items():
            ratio = val / overall_median
            for factor in standard_factors:
                if 0.85 * factor <= ratio <= 1.15 * factor:
                    return {
                        "present": True,
                        "column": col,
                        "group_value": str(grp_val),
                        "factor": factor,
                    }
                inv_factor = 1.0 / factor
                if 0.85 * inv_factor <= ratio <= 1.15 * inv_factor:
                    return {
                        "present": True,
                        "column": col,
                        "group_value": str(grp_val),
                        "factor": inv_factor,
                    }

    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(df: pd.DataFrame) -> dict[str, Any]:
    """Detect sentinel replacement values standing in for missing observations.

    Meaning:
        A numeric field contains arbitrary marker numbers like -1, -999, 999, or 9999
        representing null or unrecorded measurements instead of leaving blanks.

    Expected Return Format (per findings.yaml):
        {
            "present": bool,
            "column": Optional[str],
            "value": Optional[int | float],
            "count": Optional[int]
        }

    Hint:
        Inspect `df[col].value_counts()` on numeric features or check for negative
        values in strictly positive domains (such as rainfall or distance).

    Gotcha:
        Make sure the sentinel value is truly impossible in the domain (e.g. rain_mm_week < 0).
        Report the exact count of rows carrying that sentinel in train.csv.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    sentinel_candidates = [-1, -999, 999, 9999, -99]

    for col in num_cols:
        if col in ["kg_per_worker", "kg_weighed_at_factory"]:
            continue
        for s_val in sentinel_candidates:
            count = int((df[col] == s_val).sum())
            if count >= 10:
                return {
                    "present": True,
                    "column": col,
                    "value": s_val,
                    "count": count,
                }

    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(df: pd.DataFrame) -> dict[str, Any]:
    """Detect duplicate record rows.

    Meaning:
        Exact identical rows accidentally repeated in the dataset.

    Expected Return Format (per findings.yaml):
        {
            "present": bool,
            "count": Optional[int]  # number of EXTRA copies
        }

    Hint:
        Use `df.duplicated().sum()` to calculate the exact count of redundant rows.

    Gotcha:
        The count in findings.yaml must represent EXTRA copies (rows that would be
        deleted upon deduplication), NOT the total number of rows involved in duplicates.
    """
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        return {"present": True, "count": dup_count}
    return {"present": False, "count": None}


def check_stuck_sensor(
    df: pd.DataFrame,
    date_col: str = "date",
    min_run_length: int = 7,
) -> dict[str, Any]:
    """Detect stuck sensor output creating invariant runs across consecutive dates.

    Meaning:
        A measuring hardware device malfunctioned and transmitted identical readings
        over consecutive recording dates.

    Expected Return Format (per findings.yaml):
        {
            "present": bool,
            "column": Optional[str],
            "start_date": Optional[str],  # YYYY-MM-DD
            "end_date": Optional[str]     # YYYY-MM-DD
        }

    Hint:
        Sort records chronologically by `date_col`, then examine runs of identical
        consecutive values using `(s != s.shift()).cumsum()`.

    Gotcha:
        Ensure the run spans across days. Multiple records on the same day can
        naturally share values (e.g. constant weather for all workers on that day).
    """
    if date_col not in df.columns:
        return {"present": False, "column": None, "start_date": None, "end_date": None}

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    num_cols = df_sorted.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        if col in ["kg_per_worker", "kg_weighed_at_factory"]:
            continue
        # Evaluate daily average or single daily readings
        daily = df_sorted.groupby(date_col)[col].mean()
        runs = (daily != daily.shift(1)).cumsum()
        run_lengths = daily.groupby(runs).size()
        if not run_lengths.empty and run_lengths.max() >= min_run_length:
            max_id = run_lengths.idxmax()
            dates = daily[runs == max_id].index
            return {
                "present": True,
                "column": col,
                "start_date": str(dates[0])[:10],
                "end_date": str(dates[-1])[:10],
            }

    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(
    df: pd.DataFrame,
    target_col: str = "kg_per_worker",
    typo_factor: float = 10.0,
) -> dict[str, Any]:
    """Detect typographical multiplier errors in the target column.

    Meaning:
        Target values mistakenly entered with an extra trailing zero (approx 10x too large),
        creating an anomalous cluster far above normal realistic yields.

    Expected Return Format (per findings.yaml):
        {
            "present": bool,
            "count": Optional[int]  # rows typed ~10x too big
        }

    Hint:
        Sort the target variable with `.sort_values()`, observe the distribution gaps,
        or compare against expected tea harvesting capacity per worker-day.

    Gotcha:
        Look for a distinct gap between the maximum normal yield (e.g. ~54 kg)
        and typographical errors (>200 kg).
    """
    if target_col not in df.columns:
        return {"present": False, "count": None}

    targets = df[target_col].dropna().sort_values()
    if len(targets) < 10:
        return {"present": False, "count": None}

    # Find the largest relative jump in the upper tail
    diffs = targets.diff()
    threshold = targets.quantile(0.95) * 3.0
    anomalous = df[df[target_col] > threshold]
    count = int(len(anomalous))

    if count > 0:
        return {"present": True, "count": count}

    return {"present": False, "count": None}


def check_spelling_variants(
    df_train: pd.DataFrame,
    df_clean: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Detect inconsistent typographical variants in categorical columns.

    Meaning:
        Text columns containing informal casing, whitespace, or abbreviated
        representations of the same underlying category level.

    Expected Return Format (per findings.yaml):
        {
            "present": bool,
            "column": Optional[str],
            "true_levels": Optional[int]  # count of canonical categories
        }

    Hint:
        Inspect `df[col].value_counts()`. Reference `later.csv` to see canonical clean levels.

    Gotcha:
        Leading/trailing whitespaces, casing differences, and abbreviation dots
        (e.g. 'fir.', 'first ') artificially inflate unique category counts.
    """
    text_cols = [c for c in df_train.select_dtypes(include=["object", "string", "str"]).columns
                 if c not in ["record_id", "date"]]

    for col in text_cols:
        train_unique = df_train[col].dropna().unique()
        # Canonical levels can be inferred from clean later.csv or stripped lowercase normalization
        if df_clean is not None and col in df_clean.columns:
            clean_levels = len(df_clean[col].dropna().unique())
            if len(train_unique) > clean_levels:
                return {
                    "present": True,
                    "column": col,
                    "true_levels": clean_levels,
                }
        else:
            normalized = df_train[col].dropna().astype(str).str.strip().str.lower()
            normalized = normalized.str.replace(r"\.", "", regex=True)
            norm_levels = len(normalized.unique())
            if len(train_unique) > norm_levels:
                return {
                    "present": True,
                    "column": col,
                    "true_levels": norm_levels,
                }

    return {"present": False, "column": None, "true_levels": None}
