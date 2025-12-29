import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from pymongo import MongoClient


st.set_page_config(page_title="NYC Taxi Anomaly Dashboard", layout="wide")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "anomalydb")
MONGO_COL = os.getenv("MONGO_COL", "predictions")


@st.cache_resource
def _mongo():
    return MongoClient(MONGO_URI)


def _fetch(limit: int = 2000, stream_id: str | None = None):
    col = _mongo()[MONGO_DB][MONGO_COL]
    q = {"kind": "stream"}
    if stream_id:
        q["stream_id"] = stream_id
    docs = list(col.find(q).sort("ts", -1).limit(limit))
    docs.reverse()
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


st.title("NYC Taxi / Time-Series Anomaly Dashboard")

with st.sidebar:
    st.subheader("Connection")
    st.text_input("Mongo URI", MONGO_URI, disabled=True)
    limit = st.slider("History points", 200, 5000, 2000, step=100)
    stream_id = st.text_input("Stream ID", "default")
    refresh = st.button("Refresh")

df = _fetch(limit=limit, stream_id=stream_id)

if df.empty:
    st.info(
        "No streaming points found in Mongo yet. "
        "Call POST /predict_point a few times to start populating the dashboard."
    )
    st.stop()

anoms = int((df["flag"] == 1).sum())
st.metric("Anomaly count (in view)", anoms)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Value (with anomaly markers)")
    st.line_chart(df.set_index("dt")["value"])
    if anoms > 0:
        st.caption("Anomalies: red dots")
        st.scatter_chart(df[df["flag"] == 1].set_index("dt")["value"])

with col2:
    st.subheader("Reconstruction error")
    st.line_chart(df.set_index("dt")["error"])
    st.caption("Tip: if you never see anomalies, lower the threshold percentile in train/config.yaml.")

st.subheader("Latest records")
st.dataframe(df[["dt", "stream_id", "value", "error", "flag", "label", "window"]].tail(50), use_container_width=True)
