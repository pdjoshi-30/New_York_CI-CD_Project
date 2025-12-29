"""Evaluate anomaly detection against NAB labeled windows.

This is a simple, local, point-wise evaluation:
  - Convert NAB anomaly windows to a boolean label per timestamp.
  - Score the full series with the LSTM autoencoder.
  - Align each window score to its end index (t = window-1 .. end).
  - Compute precision, recall, F1.

Note: NAB's official scoring uses a more nuanced scoring function.
This script is meant to give an easy, local sanity-check metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from app.model import ModelWrapper
from app.config import Settings


def load_nyc_taxi_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # NAB's nyc_taxi.csv is timestamp,value
    if "timestamp" not in df.columns or "value" not in df.columns:
        df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df["value"] = df["value"].astype(float)
    return df


def load_windows(labels_json: str, key: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    data = json.loads(Path(labels_json).read_text())
    if key not in data:
        # try common variants
        for k in data.keys():
            if k.endswith("nyc_taxi.csv"):
                key = k
                break
    windows = []
    for start, end in data.get(key, []):
        windows.append((pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)))
    return windows


def windows_to_labels(ts: pd.Series, windows: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> np.ndarray:
    y = np.zeros(len(ts), dtype=int)
    for s, e in windows:
        y[(ts >= s) & (ts <= e)] = 1
    return y


def prf1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main(args):
    df = load_nyc_taxi_csv(args.csv)
    windows = load_windows(args.labels, args.key)
    y_true = windows_to_labels(df["timestamp"], windows)

    settings = Settings()
    mw = ModelWrapper(settings.model_dir, settings.model_name, settings.scaler_name, settings.threshold_name)
    if not mw.is_loaded():
        raise SystemExit("Model not loaded. Train first: python train/train.py --config train/config.yaml")

    window = mw.default_window()
    errors, flags, end_idx = mw.score_series(df["value"].to_numpy(), window=window)

    # Build point-wise prediction array: we only have predictions from end_idx onward.
    y_pred = np.zeros(len(df), dtype=int)
    for f, i in zip(flags, end_idx):
        y_pred[int(i)] = int(f)

    # Ignore initial warmup where we cannot score
    valid = np.zeros(len(df), dtype=bool)
    valid[window - 1 :] = True
    metrics = prf1(y_true[valid], y_pred[valid])

    print("Series:", args.csv)
    print("Label windows:", len(windows))
    print("Window size:", window)
    print("Threshold:", float(mw.threshold))
    print("Metrics (point-wise, excluding warmup):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/nyc_taxi.csv")
    p.add_argument("--labels", default="data/nab_windows.json")
    p.add_argument("--key", default="realKnownCause/nyc_taxi.csv")
    args = p.parse_args()
    main(args)
