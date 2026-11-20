"""Runner script to execute all 7 data checks on MP055 training data.

Loads configurations from params.yaml and reports diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path
import yaml

from src.mp055.data_checks import (
    check_duplicate_rows,
    check_leaking_column,
    check_sentinel_value,
    check_spelling_variants,
    check_stuck_sensor,
    check_target_typos,
    check_unit_mixup,
)
from src.mp055.io_utils import load_csv, load_params, sha256_of


def run_all_checks(save_report: bool = True) -> dict[str, dict]:
    """Execute all seven data checks on train.csv and print a diagnostic report.

    Args:
        save_report: If True, writes the data_issues section to reports/findings.yaml.

    Returns:
        dict: Summary dictionary of issues mapped to findings.yaml format.
    """
    params = load_params("params.yaml")
    train_path = params["data"]["raw_train_path"]
    later_path = params["data"]["raw_later_path"]
    holdout_path = params["data"]["raw_holdout_path"]
    expected_sha = params["data"]["train_sha256"]

    # 1. Integrity check
    actual_sha = sha256_of(train_path)
    print("=" * 60)
    print("MP055 DATA QUALITY AUDIT - INDORE TEA GROWERS' ESTATE")
    print("=" * 60)
    print(f"Dataset path       : {train_path}")
    print(f"Computed SHA-256   : {actual_sha}")
    print(f"Config SHA-256     : {expected_sha}")
    if actual_sha.lower() != expected_sha.lower():
        print("ERROR: SHA-256 hash mismatch! Data file might have been modified.")
        sys.exit(1)
    else:
        print("SHA-256 verification: PASSED (Byte-for-byte match)")
    print("-" * 60)

    # 2. Load datasets safely
    df_train = load_csv(train_path)
    df_later = load_csv(later_path) if Path(later_path).exists() else None
    df_holdout = load_csv(holdout_path) if Path(holdout_path).exists() else None

    # 3. Run individual checks
    results = {}

    print("Executing Stage 1 Data Checks...")

    # Leaking column
    try:
        results["leaking_column"] = check_leaking_column(df_train, df_holdout)
        print(f" [1/7] leaking_column     : {results['leaking_column']}")
    except NotImplementedError:
        print(" [1/7] leaking_column     : NOT IMPLEMENTED")

    # Unit mixup
    try:
        results["unit_mixup"] = check_unit_mixup(df_train)
        print(f" [2/7] unit_mixup         : {results['unit_mixup']}")
    except NotImplementedError:
        print(" [2/7] unit_mixup         : NOT IMPLEMENTED")

    # Sentinel value
    try:
        results["sentinel_value"] = check_sentinel_value(df_train)
        print(f" [3/7] sentinel_value     : {results['sentinel_value']}")
    except NotImplementedError:
        print(" [3/7] sentinel_value     : NOT IMPLEMENTED")

    # Duplicate rows
    try:
        results["duplicate_rows"] = check_duplicate_rows(df_train)
        print(f" [4/7] duplicate_rows     : {results['duplicate_rows']}")
    except NotImplementedError:
        print(" [4/7] duplicate_rows     : NOT IMPLEMENTED")

    # Stuck sensor
    try:
        results["stuck_sensor"] = check_stuck_sensor(df_train)
        print(f" [5/7] stuck_sensor       : {results['stuck_sensor']}")
    except NotImplementedError:
        print(" [5/7] stuck_sensor       : NOT IMPLEMENTED")

    # Target typos
    try:
        results["target_typos"] = check_target_typos(df_train)
        print(f" [6/7] target_typos       : {results['target_typos']}")
    except NotImplementedError:
        print(" [6/7] target_typos       : NOT IMPLEMENTED")

    # Spelling variants
    try:
        results["spelling_variants"] = check_spelling_variants(df_train, df_later)
        print(f" [7/7] spelling_variants  : {results['spelling_variants']}")
    except NotImplementedError:
        print(" [7/7] spelling_variants  : NOT IMPLEMENTED")

    print("-" * 60)

    # 4. Save to reports/findings.yaml
    if save_report:
        template_path = Path("findings.yaml")
        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                findings = yaml.safe_load(f)
        else:
            findings = {"group_id": "G189", "problem_id": "MP055", "data_issues": {}}

        findings["data_issues"] = results

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_out = reports_dir / "findings.yaml"

        with open(report_out, "w", encoding="utf-8") as f:
            yaml.safe_dump(findings, f, sort_keys=False)
        print(f"Updated findings saved to {report_out}")

        # Also keep root findings.yaml synchronized
        with open(template_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(findings, f, sort_keys=False)
        print(f"Synchronized root findings saved to {template_path}")

    print("=" * 60)
    return results


if __name__ == "__main__":
    run_all_checks()
