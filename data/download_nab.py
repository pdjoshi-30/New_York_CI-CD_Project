"""Download a chosen NAB file (nyc_taxi.csv) from the NAB GitHub repo."""
import argparse
import requests
import os

URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"

def download(out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    print("Downloaded", out_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/nyc_taxi.csv")
    args = p.parse_args()
    download(args.out)
