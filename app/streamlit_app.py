"""Web app to test the model: pick or upload a ride, see the signal, call the API.

API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", Path(__file__).parent.parent / "samples"))
AXES = ["ax", "ay", "az", "rx", "ry", "rz"]

st.set_page_config(page_title="Ride crash detection", page_icon="🛴", layout="wide")
st.title("Ride crash detection")
st.caption("Send the IMU recording of a ride to the API and get the probability that it contains a crash.")

with st.sidebar:
    st.header("API")
    api_url = st.text_input("URL", API_URL)
    try:
        info = requests.get(f"{api_url}/model", timeout=5).json()
        st.success(f"Model {info['model_name']} v{info['model_version']}")
        st.write(f"Test ROC AUC: {info.get('test_roc_auc')}")
        st.write(f"Decision threshold: {info['threshold']}")
    except Exception:
        st.error("API unreachable")
    st.markdown(f"[Swagger documentation]({api_url}/docs)")

samples = {p.stem: p for p in sorted(SAMPLES_DIR.glob("*.json"))}
source = st.radio("Ride", ["Example ride", "Upload a CSV"], horizontal=True)

ride = None
if source == "Example ride":
    if samples:
        name = st.selectbox("Example (real rides from the test set, never seen in training)", list(samples))
        ride = pd.DataFrame(json.loads(samples[name].read_text()))
    else:
        st.warning("No example ride found.")
else:
    uploaded = st.file_uploader(
        "CSV with the columns ax, ay, az, rx, ry, rz (one row per sample)", type="csv"
    )
    if uploaded:
        frame = pd.read_csv(uploaded)
        missing = [axis for axis in AXES if axis not in frame.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            ride = frame[AXES]

if ride is not None:
    signal = pd.DataFrame(
        {
            "acceleration": (ride[["ax", "ay", "az"]].astype(float) ** 2).sum(axis=1) ** 0.5,
            "rotation": (ride[["rx", "ry", "rz"]].astype(float) ** 2).sum(axis=1) ** 0.5,
        }
    )
    left, right = st.columns(2)
    left.subheader("Acceleration magnitude")
    left.line_chart(signal["acceleration"])
    right.subheader("Rotation magnitude")
    right.line_chart(signal["rotation"])

    if st.button("Predict", type="primary"):
        payload = {axis: [None if pd.isna(v) else float(v) for v in ride[axis]] for axis in AXES}
        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            verdict = "Crash detected" if result["crash_detected"] else "No crash detected"
            (st.error if result["crash_detected"] else st.success)(verdict)
            a, b, c = st.columns(3)
            a.metric("Crash probability", f"{result['crash_probability']:.1%}")
            b.metric("Samples", result["n_samples"])
            c.metric("Model version", result["model_version"])
            st.progress(result["crash_probability"])
        except requests.HTTPError:
            st.error(f"API error {response.status_code}: {response.text[:300]}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
