"""Unit tests: feature engineering."""

import numpy as np
import pandas as pd

from crashdetect.features import AXES, FEATURE_NAMES, features_frame, ride_features
from tests.conftest import make_ride


def test_feature_names_are_stable():
    ride = make_ride(np.random.default_rng(1), crash=False)
    assert list(ride_features(ride)) == FEATURE_NAMES
    assert features_frame(ride).shape == (1, len(FEATURE_NAMES))


def test_known_values():
    ride = pd.DataFrame({axis: [0.0] * 20 for axis in AXES})
    ride["az"] = 2.0
    feats = ride_features(ride)
    assert feats["acc_mean"] == 2.0
    assert feats["acc_std"] == 0.0
    assert feats["jerk_max"] == 0.0
    assert feats["n_samples"] == 20
    assert feats["missing_share"] == 0.0


def test_missing_values_are_ignored():
    ride = make_ride(np.random.default_rng(2), crash=False)
    ride.loc[:9, "ax"] = np.nan
    feats = ride_features(ride)
    assert np.isfinite(feats["ax_mean"])
    assert 0 < feats["missing_share"] < 1


def test_fully_missing_ride_does_not_crash():
    ride = pd.DataFrame({axis: [None] * 15 for axis in AXES})
    feats = ride_features(ride)
    assert np.isnan(feats["acc_max"])
    assert feats["missing_share"] == 1.0


def test_crash_has_higher_rotation_peak():
    rng = np.random.default_rng(3)
    assert ride_features(make_ride(rng, True))["gyr_max"] > ride_features(make_ride(rng, False))["gyr_max"]
