"""Shared configuration: parameters file, MLflow names."""

import os
from pathlib import Path

import yaml

MODEL_NAME = "crash-detector"
EXPERIMENT_NAME = "crash-detection"
PRODUCTION_ALIAS = "production"
REPORTS_DIR = Path("reports")


def tracking_uri() -> str:
    """Local SQLite store by default; override with MLFLOW_TRACKING_URI."""
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def load_params(path: str | Path = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
