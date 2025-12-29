# IoT Anomaly — Production-Ready Demo
Real-time anomaly detection on IoT-like time series using a PyTorch LSTM Autoencoder trained on the **Numenta Anomaly Benchmark (NAB)** `nyc_taxi.csv` sample.

This repo is built with production-quality structure:
- config via pydantic
- logging
- model checkpoints and scaler persistence
- FastAPI microservice for predictions
- Dockerfile + docker-compose for local demo (includes MongoDB)
- GitHub Actions CI/CD workflow stub

## Quickstart (local)

1. Create a Python venv and install:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Download dataset (script):
   ```
   python data/download_nab.py --out data/nyc_taxi.csv
   ```

3. Train model:
   ```
   python train/train.py --config train/config.yaml
   ```

4. Start MongoDB (docker) and API:
   ```
   docker compose -f infra/docker-compose.yml up --build
   ```

5. Stream points (this is what dashboards typically do):
   ```
   curl -X POST http://localhost:8000/predict_point \
     -H "Content-Type: application/json" \
     -d '{"stream_id":"default","value":23.1}'
   ```

   **Note:** you’ll see `label=warmup` until the model has received `window_size` points.

6. Optional dashboard (Streamlit):
   ```
   pip install -r requirements.txt -r requirements-dashboard.txt
   streamlit run dashboard/streamlit_app.py
   ```

See `train/`, `app/` and `infra/` folders for more details.
