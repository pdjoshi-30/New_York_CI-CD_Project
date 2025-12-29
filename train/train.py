"""Train LSTM Autoencoder for univariate time series anomaly detection (nyc_taxi)."""
import os
import argparse
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from common.model_arch import LSTMAutoencoder

log = logging.getLogger("train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class WindowDataset(Dataset):
    def __init__(self, series: np.ndarray, window: int):
        self.series = series
        self.window = window
        self.n = len(series) - window + 1

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        w = self.series[idx: idx + self.window]
        return torch.tensor(w, dtype=torch.float32).unsqueeze(-1)  # shape (window, 1)

 

def load_series(path: str, value_col: str="value"):
    df = pd.read_csv(path)
    if value_col not in df.columns:
        df.columns = ["timestamp", "value"]
    series = df["value"].astype(float).to_numpy()
    return series

def train(cfg: dict):
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["training"]
    m_cfg = cfg["model"]
    out_cfg = cfg["output"]
    threshold_cfg = cfg.get("threshold", {})
    Path(out_cfg["model_dir"]).mkdir(parents=True, exist_ok=True)

    series = load_series(ds_cfg["path"], ds_cfg.get("value_column", "value"))
    n = len(series)
    val_n = int(n * tr_cfg["val_split"])
    train_series = series[:-val_n]
    val_series = series[-val_n:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_series.reshape(-1,1)).ravel()
    val_scaled = scaler.transform(val_series.reshape(-1,1)).ravel()

    # windowing
    window = tr_cfg["window_size"]
    train_ds = WindowDataset(train_scaled, window)
    val_ds = WindowDataset(val_scaled, window)
    train_loader = DataLoader(train_ds, batch_size=tr_cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tr_cfg["batch_size"], shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(
        input_size=1,
        hidden_size=m_cfg["hidden_size"],
        latent_size=m_cfg["latent_size"],
        num_layers=m_cfg["num_layers"],
    ).to(device)
    #optimizer = optim.Adam(model.parameters(), lr=tr_cfg["lr"])
    optimizer = optim.Adam(model.parameters(), lr=float(tr_cfg["lr"]))

    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, tr_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_train = total_loss / (len(train_loader.dataset) or 1)
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * batch.size(0)
        avg_val = val_loss / (len(val_loader.dataset) or 1)
        log.info("Epoch %d train_loss=%.6f val_loss=%.6f", epoch, avg_train, avg_val)
        # checkpoint
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(out_cfg["model_dir"], out_cfg["model_name"]))
            # save scaler
            np.savez_compressed(os.path.join(out_cfg["model_dir"], out_cfg["scaler_name"]), mean=scaler.mean_, scale=scaler.scale_)
            log.info("Saved best model, val_loss=%.6f", best_val)

    # compute threshold on reconstruction error over training windows
    model.load_state_dict(torch.load(os.path.join(out_cfg["model_dir"], out_cfg["model_name"]), map_location=device))
    model.to(device)
    model.eval()
    # compute errors on train set
    train_loader_full = DataLoader(WindowDataset(train_scaled, window), batch_size=tr_cfg["batch_size"], shuffle=False)
    errors = []
    with torch.no_grad():
        for batch in train_loader_full:
            batch = batch.to(device)
            recon = model(batch)
            err = ((recon - batch)**2).mean(dim=(1,2)).cpu().numpy()
            errors.extend(err.tolist())
    errors = np.array(errors)
    perc = threshold_cfg.get("percentile", 99.0)
    threshold = float(np.percentile(errors, perc))
    # save threshold
    with open(os.path.join(out_cfg["model_dir"], "threshold.txt"), "w") as f:
        f.write(str(threshold))
    log.info("Threshold (percentile=%.2f) = %.6f", perc, threshold)

    # persist metadata to make inference robust (no hard-coded architecture in the API)
    meta = {
        "window_size": int(window),
        "model": {
            "hidden_size": int(m_cfg["hidden_size"]),
            "latent_size": int(m_cfg["latent_size"]),
            "num_layers": int(m_cfg["num_layers"]),
        },
        "threshold": {"percentile": float(perc), "value": float(threshold)},
        "dataset": {"path": ds_cfg.get("path"), "value_column": ds_cfg.get("value_column", "value")},
        "training": {"epochs": int(tr_cfg["epochs"]), "lr": float(tr_cfg["lr"]), "val_split": float(tr_cfg["val_split"])},
    }
    with open(os.path.join(out_cfg["model_dir"], "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved model metadata to %s", os.path.join(out_cfg["model_dir"], "model_meta.json"))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="train/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg)
