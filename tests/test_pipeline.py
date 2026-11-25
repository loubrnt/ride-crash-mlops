"""Integration tests: data stages, training, MLflow registry, evaluation gate, export."""

import json

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from crashdetect import data, evaluate
from crashdetect.config import MODEL_NAME, PRODUCTION_ALIAS
from crashdetect.features import FEATURE_NAMES
from crashdetect.train import build_model


def test_build_features_one_row_per_ride(raw):
    features = data.build_features(raw)
    assert len(features) == raw["ride_id"].nunique() == 100
    assert features["label"].sum() == 20
    assert "topple" not in features.columns  # the crash marker must never become a feature
    crash = features[features["label"] == 1]
    assert (crash["group_id"] == crash["ride_id"]).all()


def test_split_keeps_groups_together(raw):
    features = data.build_features(raw)
    train, test = data.split_features(features, test_share=0.25, random_state=0)
    assert len(train) + len(test) == len(features)
    assert set(train["group_id"]).isdisjoint(test["group_id"])


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_model({"model": "nope", "random_state": 0})


def test_training_registers_a_version(project):
    report = project["reports"]["train"]
    assert report["model_version"] == "1"
    run = mlflow.get_run(report["run_id"])
    assert run.data.params["train.model"] == "random_forest"
    assert "cv_roc_auc" in run.data.metrics


def test_model_beats_random(project):
    assert project["reports"]["eval"]["test_roc_auc"] > 0.7


def test_gate_promotes_to_production(project):
    assert project["reports"]["eval"]["gate_passed"] is True
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    assert str(version.version) == project["reports"]["train"]["model_version"]


def test_gate_fails_when_bar_is_too_high(project):
    params = json.loads(json.dumps(project["params"]))
    params["gate"]["min_auc"] = 1.01
    assert evaluate.evaluate(params)["gate_passed"] is False


def test_exported_model_predicts(project):
    model = mlflow.sklearn.load_model(str(project["root"] / "model"))
    test = pd.read_csv(project["root"] / "data" / "test.csv")
    proba = model.predict_proba(test[FEATURE_NAMES])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert project["reports"]["export"]["model_version"] == "1"
