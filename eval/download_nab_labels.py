"""Download NAB anomaly windows JSON.

NAB provides labeled anomaly windows for each time series.
This script downloads the official mapping so you can evaluate locally.
"""

from __future__ import annotations

import argparse
import os

import requests


URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"


def download(out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"Downloaded labels to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/nab_windows.json")
    args = p.parse_args()
    download(args.out)
