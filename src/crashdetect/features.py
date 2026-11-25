"""Feature engineering: one ride (IMU time series) -> one row of summary statistics.

The same function is used for training and inside the API, so there is no
training/serving skew. The `topple` column of the raw data is never read here:
it marks the crash itself and would leak the label.
"""

import warnings

import numpy as np
import pandas as pd

AXES = ["ax", "ay", "az", "rx", "ry", "rz"]
SIGNALS = [*AXES, "acc", "gyr", "jerk", "gyr_diff"]
STATS = ["mean", "std", "min", "max", "p01", "p99"]
FEATURE_NAMES = [f"{s}_{k}" for s in SIGNALS for k in STATS] + ["n_samples", "missing_share"]


def _stats(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [float("nan")] * len(STATS)
    p01, p99 = np.percentile(x, [1, 99])
    return [float(x.mean()), float(x.std()), float(x.min()), float(x.max()), float(p01), float(p99)]


def ride_features(ride: pd.DataFrame) -> dict[str, float]:
    """Summary statistics of one ride. `ride` has one row per sample and the 6 IMU columns."""
    values = {axis: pd.to_numeric(ride[axis], errors="coerce").to_numpy(dtype=float) for axis in AXES}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values["acc"] = np.sqrt(values["ax"] ** 2 + values["ay"] ** 2 + values["az"] ** 2)
        values["gyr"] = np.sqrt(values["rx"] ** 2 + values["ry"] ** 2 + values["rz"] ** 2)
    values["jerk"] = np.abs(np.diff(values["acc"]))
    values["gyr_diff"] = np.abs(np.diff(values["gyr"]))

    out: dict[str, float] = {}
    for signal in SIGNALS:
        for stat, value in zip(STATS, _stats(values[signal]), strict=True):
            out[f"{signal}_{stat}"] = value
    n = len(ride)
    out["n_samples"] = float(n)
    out["missing_share"] = float(np.isnan(values["acc"]).mean()) if n else 1.0
    return out


def features_frame(ride: pd.DataFrame) -> pd.DataFrame:
    """One-row DataFrame in the column order expected by the model."""
    return pd.DataFrame([ride_features(ride)], columns=FEATURE_NAMES)
