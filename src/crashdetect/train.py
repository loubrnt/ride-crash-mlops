"""Training stage: fit the model on the train set, track everything in MLflow.

Each run logs parameters, cross-validated metrics and the model, and registers a
new version in the MLflow Model Registry. Promotion to the `production` alias is
NOT done here: that is the job of the evaluation gate (crashdetect.evaluate).
"""

import json
import subprocess

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from crashdetect.config import EXPERIMENT_NAME, MODEL_NAME, REPORTS_DIR, load_params, tracking_uri
from crashdetect.features import FEATURE_NAMES
from crashdetect.metrics import classification_metrics

TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def build_model(p: dict) -> Pipeline:
    """Imputation is part of the model: rides with missing sensor values are still scored."""
    if p["model"] == "logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=p["random_state"]),
        )
    if p["model"] == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                class_weight="balanced",
                random_state=p["random_state"],
                n_jobs=-1,
            ),
        )
    raise ValueError(f"unknown model: {p['model']}")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train(params: dict) -> dict:
    p_data, p_train = params["data"], params["train"]
    train_df = pd.read_csv(p_data["train_path"])
    X, y, groups = train_df[FEATURE_NAMES], train_df["label"], train_df["group_id"]

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=p_train["model"]) as run:
        mlflow.set_tags({"git_commit": git_commit(), "stage": "train"})
        mlflow.log_params({f"train.{k}": v for k, v in p_train.items()})
        mlflow.log_params({"data.test_share": p_data["test_share"], "data.n_train": len(train_df)})

        # Out-of-fold predictions on the train set: an honest estimate before touching the test set
        cv = GroupKFold(n_splits=p_train["cv_folds"])
        oof = cross_val_predict(build_model(p_train), X, y, groups=groups, cv=cv, method="predict_proba")[
            :, 1
        ]
        cv_metrics = {f"cv_{k}": v for k, v in classification_metrics(y, oof, p_train["threshold"]).items()}
        mlflow.log_metrics(cv_metrics)

        model = build_model(p_train).fit(X, y)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X.head(5), model.predict_proba(X.head(5))[:, 1]),
            input_example=X.head(2),
            registered_model_name=MODEL_NAME,
            # skops refuses unknown types by default; these two are the tree storage of our own model
            skops_trusted_types=TRUSTED_TYPES,
        )
        report = {
            "run_id": run.info.run_id,
            "model_version": str(info.registered_model_version),
            "model": p_train["model"],
            **{k: round(v, 4) for k, v in cv_metrics.items()},
        }

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "train.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(train(load_params()), indent=2))
