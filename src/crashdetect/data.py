"""Data stages of the pipeline.

    python -m crashdetect.data features   raw parquet -> one row of features per ride
    python -m crashdetect.data split      features -> train / test, grouped by crash
    python -m crashdetect.data samples    a few test rides -> samples/*.json (web app demo)

Label: a ride is a crash (1) when `crash_ride_id` is null. Baseline rides (0) point
to the crash ride they were matched with (4 baselines per crash).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit

from crashdetect.config import load_params
from crashdetect.features import AXES, FEATURE_NAMES, ride_features

META = ["ride_id", "group_id", "city", "label"]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per ride: identifiers, label and features."""
    rows = []
    for ride_id, ride in raw.groupby("ride_id", sort=True, observed=True):
        crash_ride_id = ride["crash_ride_id"].iloc[0]
        is_crash = pd.isna(crash_ride_id)
        rows.append(
            {
                "ride_id": ride_id,
                "group_id": ride_id if is_crash else crash_ride_id,
                "city": int(ride["city"].iloc[0]),
                "label": int(is_crash),
                **ride_features(ride.sort_values("time_index")),
            }
        )
    return pd.DataFrame(rows, columns=META + FEATURE_NAMES)


def split_features(features: pd.DataFrame, test_share: float, random_state: int):
    """Grouped split: a crash ride and its baselines never end up on both sides."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=random_state)
    train_idx, test_idx = next(splitter.split(features, features["label"], groups=features["group_id"]))
    return features.iloc[train_idx], features.iloc[test_idx]


def read_raw(path: str, ride_ids: list[str] | None = None) -> pd.DataFrame:
    columns = ["ride_id", "crash_ride_id", "city", "time_index", *AXES]
    filters = [("ride_id", "in", ride_ids)] if ride_ids else None
    return pq.read_table(path, columns=columns, filters=filters).to_pandas()


def stage_features(p: dict) -> None:
    features = build_features(read_raw(p["raw_path"]))
    Path(p["features_path"]).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(p["features_path"], index=False, float_format="%.6g")
    print(f"{len(features)} rides, {int(features['label'].sum())} crashes -> {p['features_path']}")


def stage_split(p: dict) -> None:
    features = pd.read_csv(p["features_path"])
    train, test = split_features(features, p["test_share"], p["random_state"])
    train.to_csv(p["train_path"], index=False)
    test.to_csv(p["test_path"], index=False)
    print(f"train: {len(train)} rides ({train['label'].mean():.1%} crashes), test: {len(test)} rides")


def stage_samples(p: dict, out_dir: str = "samples") -> None:
    """Export a few test rides (half crashes, half safe) as JSON payloads for the API."""
    test = pd.read_csv(p["test_path"])
    half = p["n_samples"] // 2
    picked = pd.concat([test[test["label"] == label].sort_values("ride_id").head(half) for label in (1, 0)])
    raw = read_raw(p["raw_path"], ride_ids=picked["ride_id"].tolist())
    Path(out_dir).mkdir(exist_ok=True)
    counters = {1: 0, 0: 0}
    for _, row in picked.iterrows():
        ride = raw[raw["ride_id"] == row["ride_id"]].sort_values("time_index")
        counters[row["label"]] += 1
        name = f"{'crash' if row['label'] else 'safe'}_{counters[row['label']]}.json"
        payload = {axis: [None if pd.isna(v) else round(float(v), 4) for v in ride[axis]] for axis in AXES}
        Path(out_dir, name).write_text(json.dumps(payload))
        print(f"{name}: {len(ride)} samples")


if __name__ == "__main__":
    stages = {"features": stage_features, "split": stage_split, "samples": stage_samples}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        sys.exit(f"usage: python -m crashdetect.data [{'|'.join(stages)}]")
    stages[sys.argv[1]](load_params()["data"])
