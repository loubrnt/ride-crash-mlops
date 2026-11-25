"""Prediction API. Swagger UI: /docs

MODEL_URI=model uvicorn crashdetect.api.main:app --port 8000
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

from crashdetect.api.schemas import Health, ModelInfo, Prediction, Ride
from crashdetect.features import features_frame

state: dict = {}


def load_model(model_uri: str) -> None:
    state["model"] = mlflow.sklearn.load_model(model_uri)
    # One request = one ride: a thread pool per prediction costs more than it saves
    # (measured in ai-infrastructure/). INFERENCE_THREADS=1 turns it off.
    threads = os.getenv("INFERENCE_THREADS")
    if threads and hasattr(state["model"][-1], "n_jobs"):
        state["model"][-1].set_params(n_jobs=int(threads))
    meta_path = Path(model_uri) / "model_meta.json"
    state["meta"] = (
        json.loads(meta_path.read_text())
        if meta_path.exists()
        else {"model_name": "crash-detector", "model_version": "unknown", "threshold": 0.5}
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(os.getenv("MODEL_URI", "model"))
    yield
    state.clear()


app = FastAPI(
    title="Ride crash detection API",
    description="Send the IMU recording of a ride, get the probability that it contains a crash.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=Health, tags=["monitoring"])
def health() -> Health:
    return Health(status="ok", model_loaded="model" in state)


@app.get("/model", response_model=ModelInfo, tags=["monitoring"])
def model_info() -> ModelInfo:
    return ModelInfo(**{k: state["meta"].get(k) for k in ModelInfo.model_fields})


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict(ride: Ride) -> Prediction:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    frame = pd.DataFrame(ride.model_dump())
    probability = float(state["model"].predict_proba(features_frame(frame))[0, 1])
    threshold = float(state["meta"]["threshold"])
    return Prediction(
        crash_probability=round(probability, 4),
        crash_detected=probability >= threshold,
        threshold=threshold,
        n_samples=len(frame),
        model_version=str(state["meta"]["model_version"]),
    )
