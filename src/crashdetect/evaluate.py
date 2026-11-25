"""Evaluation gate: score the freshly trained version on the held-out test set.

If ROC AUC >= gate.min_auc the version gets the `production` alias in the MLflow
Model Registry. Otherwise the script exits with code 1, which stops the pipeline
(DVC locally, GitHub Actions in CI) before any image is built.
"""

import json
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient

from crashdetect.config import MODEL_NAME, PRODUCTION_ALIAS, REPORTS_DIR, load_params, tracking_uri
from crashdetect.features import FEATURE_NAMES
from crashdetect.metrics import classification_metrics


def evaluate(params: dict) -> dict:
    train_report = json.loads((REPORTS_DIR / "train.json").read_text())
    version = train_report["model_version"]
    test = pd.read_csv(params["data"]["test_path"])

    mlflow.set_tracking_uri(tracking_uri())
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    proba = model.predict_proba(test[FEATURE_NAMES])[:, 1]
    metrics = classification_metrics(test["label"], proba, params["train"]["threshold"])

    min_auc = params["gate"]["min_auc"]
    passed = metrics["roc_auc"] >= min_auc
    with mlflow.start_run(run_id=train_report["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.set_tag("gate", "passed" if passed else "failed")
    if passed:
        MlflowClient().set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version)

    report = {
        "model_version": version,
        "n_test": len(test),
        "crash_share": round(float(test["label"].mean()), 4),
        **{f"test_{k}": round(v, 4) for k, v in metrics.items()},
        "gate_min_auc": min_auc,
        "gate_passed": passed,
    }
    (REPORTS_DIR / "eval.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = evaluate(load_params())
    print(json.dumps(result, indent=2))
    if not result["gate_passed"]:
        sys.exit(f"GATE FAILED: ROC AUC {result['test_roc_auc']} < {result['gate_min_auc']}")
