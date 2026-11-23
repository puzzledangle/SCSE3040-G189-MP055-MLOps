"""Model training and MLflow tracking pipeline for MP055.

Trains regression models on cleaned tea plucking data, tracks runs
with MLflow (backed by SQLite), registers the best model (v1), and sets
the @champion alias.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from src.mp055.clean import clean_dataset
from src.mp055.io_utils import load_csv, load_params, sha256_of

# Silence agent tracing hint from MLflow
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def get_feature_names() -> tuple[list[str], list[str], str]:
    """Return feature column subsets and target column name."""
    num_features = [
        "worker_exp_yr",
        "rain_mm_week",
        "slope_deg",
        "temp_c",
        "bush_age_yr",
        "hours_worked",
    ]
    cat_features = ["flush", "section"]
    target = "kg_per_worker"
    return num_features, cat_features, target


def build_pipeline(model: Any, num_features: list[str], cat_features: list[str]) -> Pipeline:
    """Build a complete scikit-learn Pipeline with preprocessing and model.

    Args:
        model: Regressor estimator.
        num_features: List of numerical feature names.
        cat_features: List of categorical feature names.

    Returns:
        Pipeline: Composed scikit-learn Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def train_models() -> Dict[str, Any]:
    """Execute model training, MLflow tracking, and model registration.

    Returns:
        Dict[str, Any]: Summary of the best run and champion model registration.
    """
    params = load_params("params.yaml")
    cleaned_train_path = Path(params["data"]["cleaned_train_path"])
    raw_train_path = Path(params["data"]["raw_train_path"])
    seed = int(params["settings"]["random_seed"])

    # 1. Compute and verify SHA256
    train_sha256 = sha256_of(raw_train_path)

    # 2. Ensure cleaned data exists
    if not cleaned_train_path.exists():
        print(f"Cleaned dataset not found at {cleaned_train_path}. Generating now...")
        raw_df = load_csv(raw_train_path)
        df_clean, _ = clean_dataset(raw_df, is_train=True)
        cleaned_train_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(cleaned_train_path, index=False)
    else:
        df_clean = load_csv(cleaned_train_path)

    num_features, cat_features, target = get_feature_names()
    features = num_features + cat_features
    X = df_clean[features]
    y = df_clean[target]

    # 3. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    # 4. Setup MLflow with SQLite
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "mp055-tea-yield"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Define candidate models (3+ distinct settings)
    candidate_settings = [
        {
            "name": "ridge_alpha_1.0",
            "model_type": "Ridge",
            "model": Ridge(alpha=1.0, random_state=seed),
            "hyperparameters": {"alpha": 1.0},
        },
        {
            "name": "ridge_alpha_10.0",
            "model_type": "Ridge",
            "model": Ridge(alpha=10.0, random_state=seed),
            "hyperparameters": {"alpha": 10.0},
        },
        {
            "name": "random_forest_depth_8",
            "model_type": "RandomForestRegressor",
            "model": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=seed),
            "hyperparameters": {"n_estimators": 100, "max_depth": 8},
        },
        {
            "name": "gradient_boosting_depth_4",
            "model_type": "GradientBoostingRegressor",
            "model": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.08, max_depth=4, random_state=seed
            ),
            "hyperparameters": {"n_estimators": 100, "learning_rate": 0.08, "max_depth": 4},
        },
    ]

    best_val_mae = float("inf")
    best_run_id = None
    best_pipe = None
    best_setting = None

    print("=" * 60)
    print("MP055 STAGE 2 - MODEL TRAINING & MLFLOW TRACKING")
    print(f"Tracking URI       : {tracking_uri}")
    print(f"Experiment Name    : {experiment_name}")
    print(f"Data SHA-256       : {train_sha256}")
    print("-" * 60)

    for setting in candidate_settings:
        run_name = setting["name"]
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Mandatory: Log SHA-256 of original training data
            mlflow.log_param("data_sha256", train_sha256)
            mlflow.log_param("model_type", setting["model_type"])
            mlflow.log_params(setting["hyperparameters"])
            mlflow.log_param("features", ",".join(features))

            # Train pipeline
            pipe = build_pipeline(setting["model"], num_features, cat_features)
            pipe.fit(X_train, y_train)

            # Predictions and evaluation
            train_preds = pipe.predict(X_train)
            val_preds = pipe.predict(X_val)

            train_mae = float(mean_absolute_error(y_train, train_preds))
            val_mae = float(mean_absolute_error(y_val, val_preds))

            # Mandatory: Log metric containing 'mae'
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("mae", val_mae)

            # Log model with cloudpickle serialization
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            print(
                f" Run [{run_name}] ID: {run_id[:8]}... | "
                f"Train MAE: {train_mae:.4f} kg | Val MAE: {val_mae:.4f} kg"
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_run_id = run_id
                best_pipe = pipe
                best_setting = setting

    print("-" * 60)
    print(f"Best Model Selected : {best_setting['name']}")
    print(f"Best Validation MAE : {best_val_mae:.4f} kg")
    print(f"Best Run ID         : {best_run_id}")

    # 5. Model Registration & @champion alias
    registered_model_name = "mp055-model"
    model_uri = f"runs:/{best_run_id}/model"
    reg_model = mlflow.register_model(model_uri, registered_model_name)
    champion_version = str(reg_model.version)

    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=champion_version,
    )
    print(
        f"Registered model '{registered_model_name}' version {champion_version} "
        f"with alias '@champion'."
    )

    # 6. Save model to models/model.joblib for production service
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = models_dir / "model.joblib"
    joblib.dump(best_pipe, joblib_path)
    print(f"Exported champion model pipeline to: {joblib_path}")

    # 7. Update reports/findings.yaml and findings.yaml tracking section
    for path in [Path("reports/findings.yaml"), Path("findings.yaml")]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                findings_data = yaml.safe_load(f) or {}
            if "tracking" not in findings_data:
                findings_data["tracking"] = {}
            findings_data["tracking"]["experiment_name"] = experiment_name
            findings_data["tracking"]["best_run_id"] = best_run_id
            findings_data["tracking"]["registered_model_name"] = registered_model_name
            findings_data["tracking"]["champion_version"] = int(champion_version)

            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(findings_data, f, sort_keys=False)
            print(f"Updated tracking metadata in: {path}")

    # 8. Update params.yaml
    params["model"]["algorithm"] = best_setting["model_type"]
    params["model"]["hyperparameters"] = best_setting["hyperparameters"]
    params["model"]["val_mae"] = round(best_val_mae, 4)
    with open("params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    print("Updated model configuration in params.yaml")

    print("=" * 60)
    return {
        "best_run_id": best_run_id,
        "best_val_mae": best_val_mae,
        "champion_version": champion_version,
        "registered_model_name": registered_model_name,
    }


if __name__ == "__main__":
    train_models()
