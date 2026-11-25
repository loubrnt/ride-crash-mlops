"""Where does the time of one /predict request go? Each step of the handler is timed in isolation.

uv run python ai-infrastructure/benchmarks/request_breakdown.py
"""

import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crashdetect.api.schemas import Ride  # noqa: E402
from crashdetect.features import features_frame  # noqa: E402

REPS = 100


def timed(fn) -> float:
    fn()
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.mean(samples)) * 1000


if __name__ == "__main__":
    model = mlflow.sklearn.load_model(str(ROOT / "model"))
    model_1thread = mlflow.sklearn.load_model(str(ROOT / "model"))
    model_1thread[-1].set_params(n_jobs=1)
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(ROOT / "ai-infrastructure/optimization/model.onnx"), providers=["CPUExecutionProvider"]
    )
    onnx_input = session.get_inputs()[0].name
    rows = []
    for path in sorted((ROOT / "samples").glob("*.json")):
        body = path.read_bytes()
        ride = Ride.model_validate_json(body)
        frame = pd.DataFrame(ride.model_dump())
        feats = features_frame(frame)
        rows.append(
            {
                "ride": path.stem,
                "samples": len(frame),
                "payload_kb": round(len(body) / 1000),
                "1_json_parse_and_validation_ms": timed(lambda b=body: Ride.model_validate_json(b)),
                "2_to_dataframe_ms": timed(lambda r=ride: pd.DataFrame(r.model_dump())),
                "3_feature_extraction_ms": timed(lambda f=frame: features_frame(f)),
                "4_model_default_threads_ms": timed(lambda x=feats: model.predict_proba(x)),
                "4_model_1_thread_ms": timed(lambda x=feats: model_1thread.predict_proba(x)),
                "4_model_onnx_ms": timed(
                    lambda x=feats: session.run(None, {onnx_input: x.to_numpy(dtype=np.float32)})
                ),
            }
        )
    table = pd.DataFrame(rows)
    mean = table.drop(columns=["ride"]).mean().round(2)
    mean["ride"] = "MEAN"
    table = pd.concat([table.round(2), mean.to_frame().T], ignore_index=True)
    table.to_csv(Path(__file__).parent / "request-breakdown-results.csv", index=False)
    print(table.T.to_string())
