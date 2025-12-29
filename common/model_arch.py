"""Shared model architecture used by both training and inference.

Keeping this in a common module avoids Docker/runtime import issues (e.g. app importing
from train.* which may not be copied into the container image).
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """LSTM autoencoder for univariate time-series reconstruction.

    Input:  (B, T, 1)
    Output: (B, T, 1)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        latent_size: int = 8,
        num_layers: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_size, latent_size)

        self.dec_fc = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        _, (h_n, _) = self.encoder(x)  # h_n: (num_layers, B, hidden)
        h_last = h_n[-1]  # (B, hidden)
        latent = self.enc_fc(h_last)  # (B, latent)
        dec_in = self.dec_fc(latent).unsqueeze(1).repeat(1, x.size(1), 1)  # (B, T, hidden)
        out, _ = self.decoder(dec_in)
        return out
