"""
This version adds /predict_point (streaming) that maintains an in-memory buffer
per stream_id until it has enough points to score.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.config import Settings
from app.model import ModelWrapper
from pymongo import MongoClient
import time, logging
from collections import deque
from typing import Deque, Dict, List, Optional
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import numpy as np


log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

settings = Settings()
model = ModelWrapper(settings.model_dir, settings.model_name, settings.scaler_name, settings.threshold_name)

# Simple per-process in-memory buffers for streaming.
# For production you’d move this to Redis or Mongo.
STREAM_BUFFERS: Dict[str, Deque[float]] = {}

app = FastAPI(title="IoT Anomaly - LSTM AE")

PRED_COUNTER = Counter("predictions_total", "Number of predictions", ["label"])
LATENCY = Histogram("predict_latency_seconds", "Latency for predict endpoint")

class PredictRequest(BaseModel):
    ts: Optional[float]
    values: List[float]  # accept a short series or single value list
    window: Optional[int] = None


class PredictPointRequest(BaseModel):
    """Streaming request: a single new value for a given stream."""
    stream_id: str = "default"
    ts: Optional[float] = None
    value: float
    window: Optional[int] = None

@app.get("/health")
def health():
    return {"status":"ok", "model_loaded": model.is_loaded()}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
@LATENCY.time()
def predict(req: PredictRequest):
    if not model.is_loaded():
        raise HTTPException(status_code=503, detail="model not loaded; train the model and place in models/")
    import numpy as np
    series = np.array(req.values, dtype=float)
    if series.ndim != 1:
        raise HTTPException(status_code=400, detail="values must be a 1-d list of floats")
    window = int(req.window or model.default_window())
    err, flags, end_idx = model.score_series(series, window=window)
    last_err = float(err[-1]) if len(err)>0 else 0.0
    last_flag = int(flags[-1]) if len(flags)>0 else 0
    label = "anomaly" if last_flag==1 else ("warmup" if len(series) < window else "normal")
    PRED_COUNTER.labels(label=label).inc()
    try:
        client = MongoClient(settings.mongo_uri)
        col = client[settings.mongo_db][settings.mongo_col]
        col.insert_one({
            "ts": float(req.ts) if req.ts is not None else time.time(),
            "kind": "batch",
            "input_len": int(len(series)),
            "window": int(window),
            "last_error": last_err,
            "last_flag": last_flag,
            "label": label,
        })
    except Exception as e:
        log.warning("mongo write failed: %s", e)
    return {"last_error": last_err, "last_flag": last_flag, "label": label}


@app.post("/predict_point")
@LATENCY.time()
def predict_point(req: PredictPointRequest):
    """Streaming anomaly scoring.

    - Keeps a ring buffer of the last `window` points per stream_id.
    - Returns label=warmup until enough points have arrived.
    """
    if not model.is_loaded():
        raise HTTPException(status_code=503, detail="model not loaded; train the model and place in models/")

    window = int(req.window or model.default_window())
    buf = STREAM_BUFFERS.get(req.stream_id)
    if buf is None or buf.maxlen != window:
        buf = deque(maxlen=window)
        STREAM_BUFFERS[req.stream_id] = buf

    buf.append(float(req.value))

    if len(buf) < window:
        label = "warmup"
        last_err = 0.0
        last_flag = 0
    else:
        last_err = model.score_window(np.array(list(buf), dtype=float))
        last_flag = int(last_err >= float(model.threshold))
        label = "anomaly" if last_flag == 1 else "normal"

    PRED_COUNTER.labels(label=label).inc()

    try:
        client = MongoClient(settings.mongo_uri)
        col = client[settings.mongo_db][settings.mongo_col]
        col.insert_one({
            "ts": float(req.ts) if req.ts is not None else time.time(),
            "kind": "stream",
            "stream_id": req.stream_id,
            "window": int(window),
            "value": float(req.value),
            "error": float(last_err),
            "flag": int(last_flag),
            "label": label,
        })
    except Exception as e:
        log.warning("mongo write failed: %s", e)

    return {
        "stream_id": req.stream_id,
        "buffer_len": len(buf),
        "window": int(window),
        "error": float(last_err),
        "flag": int(last_flag),
        "label": label,
    }
