"""Benchmark of the crash-detection model: sizing, baseline and optimisations.

    uv run python ai-infrastructure/benchmarks/benchmark.py            # in-process benchmarks
    uv run python ai-infrastructure/benchmarks/benchmark.py --api http://localhost:8000   # + HTTP benchmark

Everything written to the CSV files next to this script is measured, never estimated.
Each runtime is measured in its own fresh Python process, so memory figures are comparable.
"""

import argparse
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OPTIM = HERE.parent / "optimization"
sys.path.insert(0, str(ROOT / "src"))

from crashdetect.config import load_params  # noqa: E402
from crashdetect.features import FEATURE_NAMES, features_frame  # noqa: E402

WARMUP, REPS, BATCH = 50, 1000, 700


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def summarize(latencies_s: list[float]) -> dict:
    ms = np.array(latencies_s) * 1000
    return {
        "mean_ms": round(float(ms.mean()), 3),
        "p50_ms": round(float(np.percentile(ms, 50)), 3),
        "p95_ms": round(float(np.percentile(ms, 95)), 3),
        "p99_ms": round(float(np.percentile(ms, 99)), 3),
    }


def time_calls(fn, reps: int) -> list[float]:
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def load_test() -> tuple[pd.DataFrame, np.ndarray]:
    test = pd.read_csv(ROOT / load_params(ROOT / "params.yaml")["data"]["test_path"])
    return test[FEATURE_NAMES], test["label"].to_numpy()


def runtime_worker(kind: str, queue) -> None:
    """Runs in a fresh process: load one runtime, benchmark it, return the numbers."""
    from sklearn.metrics import roc_auc_score

    X, y = load_test()
    before = rss_mb()
    if kind.startswith("sklearn"):
        import mlflow

        model = mlflow.sklearn.load_model(str(ROOT / "model"))
        if kind == "sklearn_1thread":
            model[-1].set_params(n_jobs=1)

        def predict(frame):
            return model.predict_proba(frame)[:, 1]

        one, many = X.iloc[[0]], X
        size_mb = (ROOT / "model" / "model.skops").stat().st_size / 1e6
    else:
        import onnxruntime as ort

        path = OPTIM / "model.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1 if kind == "onnx_1thread" else 0
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name

        def predict(array):
            return session.run(None, {name: array})[1][:, 1]

        many = X.to_numpy(dtype=np.float32)
        one = many[:1]
        size_mb = path.stat().st_size / 1e6

    proba = predict(many)
    for _ in range(WARMUP):
        predict(one)
    single = time_calls(lambda: predict(one), REPS)
    batch = time_calls(lambda: predict(many), 30)
    queue.put(
        {
            "runtime": kind,
            "model_file_mb": round(size_mb, 2),
            **{f"batch1_{k}": v for k, v in summarize(single).items()},
            "batch1_throughput_rps": round(1 / float(np.mean(single)), 1),
            f"batch{BATCH}_mean_ms": round(float(np.mean(batch)) * 1000, 2),
            f"batch{BATCH}_throughput_rides_s": round(BATCH / float(np.mean(batch))),
            "process_rss_mb": round(rss_mb(), 1),
            "rss_added_by_runtime_mb": round(rss_mb() - before, 1),
            "test_roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "proba": proba.tolist(),
        }
    )


def export_onnx() -> None:
    import mlflow
    from skl2onnx import to_onnx

    model = mlflow.sklearn.load_model(str(ROOT / "model"))
    X, _ = load_test()
    onnx_model = to_onnx(model, X.to_numpy(dtype=np.float32)[:1], options={"zipmap": False}, target_opset=17)
    OPTIM.mkdir(exist_ok=True)
    (OPTIM / "model.onnx").write_bytes(onnx_model.SerializeToString())


def bench_runtimes() -> pd.DataFrame:
    export_onnx()
    rows = []
    for kind in ["sklearn_default", "sklearn_1thread", "onnx_default", "onnx_1thread"]:
        queue = mp.get_context("spawn").Queue()
        process = mp.get_context("spawn").Process(target=runtime_worker, args=(kind, queue))
        process.start()
        rows.append(queue.get())
        process.join()
    reference = np.array(rows[0]["proba"])
    for row in rows:
        proba = np.array(row.pop("proba"))
        row["max_abs_proba_diff_vs_baseline"] = round(float(np.abs(proba - reference).max()), 6)
        row["decisions_changed_vs_baseline"] = int(((proba >= 0.5) != (reference >= 0.5)).sum())
    return pd.DataFrame(rows)


def bench_features() -> dict:
    """Feature extraction on the 4 real example rides: the step that runs before the model in the API."""
    rides = [pd.DataFrame(json.loads(p.read_text())) for p in sorted((ROOT / "samples").glob("*.json"))]
    for ride in rides:
        features_frame(ride)
    latencies = [t for ride in rides for t in time_calls(lambda r=ride: features_frame(r), 100)]
    return {
        "step": "feature_extraction_per_ride",
        "mean_samples_per_ride": int(np.mean([len(r) for r in rides])),
        **summarize(latencies),
    }


def bench_training() -> dict:
    from crashdetect.train import build_model

    params = load_params(ROOT / "params.yaml")
    train = pd.read_csv(ROOT / params["data"]["train_path"])
    X, y = train[FEATURE_NAMES], train["label"]
    before = rss_mb()
    durations = []
    for _ in range(5):
        t0 = time.perf_counter()
        build_model(params["train"]).fit(X, y)
        durations.append(time.perf_counter() - t0)
    return {
        "step": "training_fit",
        "rows": len(X),
        "features": X.shape[1],
        "mean_s": round(float(np.mean(durations)), 2),
        "min_s": round(float(np.min(durations)), 2),
        "max_s": round(float(np.max(durations)), 2),
        "rss_added_mb": round(rss_mb() - before, 1),
    }


def bench_api(url: str) -> list[dict]:
    """End-to-end HTTP benchmark on the running container, with the real example rides."""
    import httpx

    payloads = [p.read_bytes() for p in sorted((ROOT / "samples").glob("*.json"))]
    headers = {"Content-Type": "application/json"}
    rows = []
    with httpx.Client(timeout=30) as client:
        for i in range(20):
            client.post(f"{url}/predict", content=payloads[i % 4], headers=headers)

        def call(i: int) -> float:
            t0 = time.perf_counter()
            response = client.post(f"{url}/predict", content=payloads[i % 4], headers=headers)
            response.raise_for_status()
            return time.perf_counter() - t0

        for concurrency, n in [(1, 200), (4, 400), (16, 800)]:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(concurrency) as pool:
                latencies = list(pool.map(call, range(n)))
            wall = time.perf_counter() - t0
            rows.append(
                {
                    "step": "api_predict_http",
                    "concurrency": concurrency,
                    "requests": n,
                    "mean_payload_kb": round(float(np.mean([len(p) for p in payloads])) / 1000),
                    **summarize(latencies),
                    "throughput_rps": round(n / wall, 1),
                }
            )
    return rows


def environment() -> dict:
    cpu = platform.processor()
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    import onnxruntime
    import sklearn

    return {
        "cpu": cpu,
        "logical_cores": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "gpu": "none used",
        "os": platform.platform(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "onnxruntime": onnxruntime.__version__,
        "warmup_calls": WARMUP,
        "repetitions_batch1": REPS,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", help="base URL of a running API, e.g. http://localhost:8000")
    parser.add_argument(
        "--api-only", metavar="LABEL", help="only run the HTTP benchmark and append it under this label"
    )
    args = parser.parse_args()

    if args.api_only:
        out = OPTIM / "api-comparison-results.csv"
        rows = pd.DataFrame([{"configuration": args.api_only, **r} for r in bench_api(args.api)])
        rows.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(rows.to_string())
        sys.exit(0)

    (HERE / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    runtimes = bench_runtimes()
    runtimes[runtimes["runtime"] == "sklearn_default"].to_csv(HERE / "baseline-results.csv", index=False)
    runtimes.to_csv(OPTIM / "comparison-results.csv", index=False)
    steps = [bench_features(), bench_training()]
    if args.api:
        steps += bench_api(args.api)
    pd.DataFrame(steps).to_csv(HERE / "pipeline-steps-results.csv", index=False)
    print(runtimes.T.to_string())
    print(pd.DataFrame(steps).T.to_string())
