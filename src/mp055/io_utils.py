"""I/O and hashing utilities for MP055 pipeline.

All source code reads configuration parameters via yaml.safe_load.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Union
import pandas as pd
import yaml


def load_params(params_path: Union[str, Path] = "params.yaml") -> dict[str, Any]:
    """Load configuration parameters from a YAML file using yaml.safe_load.

    Args:
        params_path: Path to the YAML file.

    Returns:
        dict: Parsed parameter dictionary.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Safely load a CSV file into a pandas DataFrame in read-only fashion.

    Args:
        path: Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p.resolve()}")
    return pd.read_csv(p)


def sha256_of(path: Union[str, Path]) -> str:
    """Compute the SHA-256 hexadecimal digest of a file.

    Args:
        path: Path to the file.

    Returns:
        str: Lowercase 64-character SHA-256 hash.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
