"""Model wrapper: loads model + scaler + threshold and exposes scoring helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from common.model_arch import LSTMAutoencoder

class ModelWrapper:
    def __init__(self, model_dir: str, model_name: str, scaler_name: str, threshold_name: str, device=None):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / model_name
        self.scaler_path = self.model_dir / scaler_name
        self.threshold_path = self.model_dir / threshold_name
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model: Optional[torch.nn.Module] = None
        self.scaler: Optional[StandardScaler] = None
        self.threshold: Optional[float] = None
        self.meta: Dict = {}
        self._load_all()

    def _load_all(self):
        if not self.model_path.exists():
            return

        meta_path = self.model_dir / "model_meta.json"
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text())
            except Exception:
                self.meta = {}
        if self.scaler_path.exists():
            data = np.load(self.scaler_path)
            scaler = StandardScaler()
            scaler.mean_ = data["mean"]
            scaler.scale_ = data["scale"]
            scaler.n_features_in_ = 1
            self.scaler = scaler

        m = (self.meta.get("model") or {})
        hidden_size = int(m.get("hidden_size", 64))
        latent_size = int(m.get("latent_size", 8))
        num_layers = int(m.get("num_layers", 1))

        model = LSTMAutoencoder(input_size=1, hidden_size=hidden_size, latent_size=latent_size, num_layers=num_layers)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model
        if self.threshold_path.exists():
            self.threshold = float(open(self.threshold_path).read().strip())

    def is_loaded(self)->bool:
        return self.model is not None and self.scaler is not None and self.threshold is not None

    def default_window(self) -> int:
        return int(self.meta.get("window_size", 30))

    def score_window(self, window_values: np.ndarray) -> float:
        """Score a single window (length = trained window). Returns MSE reconstruction error."""
        if not self.is_loaded():
            raise RuntimeError("model not loaded")
        w = np.asarray(window_values, dtype=float).reshape(-1, 1)
        w_scaled = self.scaler.transform(w).astype(np.float32)
        batch = torch.tensor(w_scaled[None, ...], dtype=torch.float32).to(self.device)  # (1, T, 1)
        with torch.no_grad():
            recon = self.model(batch)
            err = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()[0]
        return float(err)

    def score_series(self, series: np.ndarray, window: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score a series by sliding a window.

        Returns:
          errors: shape (num_windows,)
          flags:  shape (num_windows,)
          end_idx: shape (num_windows,) mapping each window score to its end index in original series

        Important:
          If len(series) < window, returns empty arrays.
        """
        if not self.is_loaded():
            raise RuntimeError("model not loaded")
        window = int(window or self.default_window())
        series = np.asarray(series, dtype=float).ravel()
        if len(series) < window:
            return np.array([]), np.array([]), np.array([])

        x_scaled = self.scaler.transform(series.reshape(-1, 1)).ravel().astype(np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(x_scaled, window_shape=window)
        batch = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1).to(self.device)  # (N, T, 1)
        with torch.no_grad():
            recon = self.model(batch)
            errors = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
        flags = (errors >= float(self.threshold)).astype(int)
        end_idx = np.arange(window - 1, window - 1 + len(errors))
        return errors, flags, end_idx
