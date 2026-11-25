"""End-to-end tests against a running API (the Docker container in CI).

    API_URL=http://localhost:8000 pytest -m e2e

They use the real example rides of samples/, so they check the real model too.
"""

import json
import os
from pathlib import Path

import httpx
import pytest

API_URL = os.getenv("API_URL")
pytestmark = [pytest.mark.e2e, pytest.mark.skipif(not API_URL, reason="API_URL not set")]
SAMPLES = sorted(Path(__file__).parent.parent.glob("samples/*.json"))


def test_health():
    assert httpx.get(f"{API_URL}/health", timeout=10).json()["model_loaded"] is True


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.stem)
def test_real_rides_are_classified(sample):
    response = httpx.post(f"{API_URL}/predict", json=json.loads(sample.read_text()), timeout=30)
    assert response.status_code == 200
    assert response.json()["crash_detected"] is sample.stem.startswith("crash")
