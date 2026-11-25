"""Export the `production` model from the MLflow Registry to ./model.

The Docker image copies this folder, so the API container is self-contained:
it needs no MLflow server at runtime.
"""

import json
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from crashdetect.config import MODEL_NAME, PRODUCTION_ALIAS, REPORTS_DIR, load_params, tracking_uri


def export(dst: str = "model") -> dict:
    mlflow.set_tracking_uri(tracking_uri())
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    shutil.rmtree(dst, ignore_errors=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}", dst_path=dst)

    meta = {
        "model_name": MODEL_NAME,
        "model_version": str(version.version),
        "run_id": version.run_id,
        "threshold": load_params()["train"]["threshold"],
    }
    eval_report = REPORTS_DIR / "eval.json"
    if eval_report.exists():
        report = json.loads(eval_report.read_text())
        if report.get("model_version") == meta["model_version"]:
            meta["test_roc_auc"] = report["test_roc_auc"]
    Path(dst, "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    print(json.dumps(export(), indent=2))
