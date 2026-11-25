"""Fixtures: a tiny synthetic dataset, so tests never need the real 268 MB file."""

import numpy as np
import pandas as pd
import pytest


def make_ride(rng: np.random.Generator, crash: bool, n: int = 300) -> pd.DataFrame:
    """Gravity on z plus noise; a crash adds a short burst of rotation and acceleration."""
    ride = pd.DataFrame(
        {
            "ax": rng.normal(0, 0.3, n),
            "ay": rng.normal(0, 0.3, n),
            "az": rng.normal(9.8, 0.3, n),
            "rx": rng.normal(0, 0.05, n),
            "ry": rng.normal(0, 0.05, n),
            "rz": rng.normal(0, 0.05, n),
        }
    )
    if crash:
        start = n // 2
        ride.loc[start : start + 5, ["rx", "ry", "rz"]] += rng.normal(4, 1, (6, 3))
        ride.loc[start : start + 5, ["ax", "ay"]] += rng.normal(8, 2, (6, 2))
    return ride


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    """Raw data in the SRV3490 layout: 20 crash rides, each with 4 baselines."""
    rng = np.random.default_rng(0)
    rides = []
    for g in range(20):
        for k in range(5):
            ride = make_ride(rng, crash=k == 0)
            ride.insert(0, "time_index", np.arange(len(ride)) * 100)
            ride.insert(0, "city", g % 8)
            ride.insert(0, "crash_ride_id", None if k == 0 else f"ride_{g}_0")
            ride.insert(0, "ride_id", f"ride_{g}_{k}")
            rides.append(ride)
    return pd.concat(rides, ignore_index=True)


@pytest.fixture(scope="session")
def project(raw, tmp_path_factory, monkeypatch_session):
    """Runs the whole pipeline once in a temp folder: features -> split -> train -> evaluate -> export."""
    from crashdetect import data, evaluate, export_model, train

    root = tmp_path_factory.mktemp("project")
    monkeypatch_session.chdir(root)
    monkeypatch_session.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{root}/mlflow.db")
    (root / "data").mkdir()
    params = {
        "data": {
            "features_path": "data/features.csv",
            "train_path": "data/train.csv",
            "test_path": "data/test.csv",
            "test_share": 0.25,
            "random_state": 0,
        },
        "train": {
            "model": "random_forest",
            "n_estimators": 20,
            "max_depth": 4,
            "cv_folds": 3,
            "random_state": 0,
            "threshold": 0.5,
        },
        "gate": {"min_auc": 0.7},
    }
    import yaml

    (root / "params.yaml").write_text(yaml.safe_dump(params))
    data.build_features(raw).to_csv(params["data"]["features_path"], index=False)
    data.stage_split(params["data"])
    reports = {"train": train.train(params), "eval": evaluate.evaluate(params)}
    reports["export"] = export_model.export(str(root / "model"))
    return {"root": root, "params": params, "reports": reports}


@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
