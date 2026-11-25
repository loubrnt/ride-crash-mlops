"""API tests with FastAPI's TestClient, on a model trained by the test pipeline."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_ride


@pytest.fixture(scope="module")
def client(project, monkeypatch_session):
    monkeypatch_session.setenv("MODEL_URI", str(project["root"] / "model"))
    from crashdetect.api.main import app

    with TestClient(app) as c:
        yield c


def payload(crash: bool, seed: int = 0) -> dict:
    return make_ride(np.random.default_rng(seed), crash=crash).to_dict(orient="list")


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model").json()
    assert body["model_name"] == "crash-detector"
    assert body["model_version"] == "1"


def test_swagger_is_served(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_predict_crash_and_safe(client):
    crash = client.post("/predict", json=payload(crash=True)).json()
    safe = client.post("/predict", json=payload(crash=False)).json()
    assert crash["crash_detected"] is True
    assert safe["crash_detected"] is False
    assert crash["crash_probability"] > safe["crash_probability"]
    assert crash["n_samples"] == 300


def test_predict_accepts_missing_values(client):
    body = payload(crash=False)
    body["ax"][0] = None
    assert client.post("/predict", json=body).status_code == 200


def test_predict_rejects_unequal_lengths(client):
    body = payload(crash=False)
    body["ax"] = body["ax"][:-1]
    assert client.post("/predict", json=body).status_code == 422


def test_predict_rejects_too_short_ride(client):
    body = {axis: [0.0] * 3 for axis in ("ax", "ay", "az", "rx", "ry", "rz")}
    assert client.post("/predict", json=body).status_code == 422


def test_inference_threads_setting(project, monkeypatch):
    from crashdetect.api import main

    monkeypatch.setenv("INFERENCE_THREADS", "1")
    main.load_model(str(project["root"] / "model"))
    assert main.state["model"][-1].n_jobs == 1
