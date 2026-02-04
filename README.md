# logistic_regression_from_scratch

A minimal-from-scratch logistic regression project (NumPy + stdlib) trained on the Raisin Classification dataset.  
It includes:
- Training entrypoint: `src/main.py`
- Lightweight MLflow-like run logging: `src/mlflow_lite.py` (writes under `mlruns/`)
- Model utilities: `src/model.py`, `src/predict.py`
- Monitoring utilities: `src/monitoring.py`
- Flask API server: `api/app.py` (endpoints: `/`, `/predict`, `/monitor`)
- Dockerfile to run the Flask API

This README explains how to run training (main), run the API locally, run via Docker, and how to reach endpoints (CLI and Python examples). It also includes common troubleshooting (Docker permission issues, container networking).

---

## Important: Python version
This repository's Dockerfile uses Python 3.12 (base image `python:3.12-slim`). For consistency, use Python 3.12 for local development too:
```bash
python --version  # should be 3.12.x
```

---

## Quick setup (local, non-Docker)

1. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
uv sync
```

2. Train the model (run the main script):
```bash
python -m src/main.py
```
What happens:
- A run directory `mlruns/run_<timestamp>/` is created by `src/mlflow_lite.start_run()`.
- Data is loaded & preprocessed.
- Model is trained with hyperparameters inside `src/main.py` (defaults: `lr=0.01`, `epochs=100`, `lambda_=0.01`).
- Baseline stats are logged to the run directory (`baseline.json`) for drift detection.
- Model artifacts are saved to `models/<run-id>/weights.npy` and `bias.npy` and copied into the run directory (`log_model`).
- Test accuracy is printed.

If you want CLI flags (learning rate, epochs, etc.), you can add `argparse` in `src/main.py` (not present in repo).

---

## API (Flask) — run locally (non-Docker)

Start the Flask API:
```bash
pip install numpy flask  # ensure deps
python api/app.py
```
By default `api/app.py` uses Flask dev server. For the container to be reachable externally it must bind to `0.0.0.0` — see Docker section / suggested change below.

Default host/port when running locally: `http://127.0.0.1:5000/`

Endpoints:
- GET `/` — health: returns `Raisin classifier API is running!`
- POST `/predict` — body: `{"features": [[...], [...]]}`. Uses the latest saved run and returns `predictions` and `probabilities`.
- POST `/monitor` — body: `{"features": [[...]], "labels": [0,1,...] (optional)}`. Returns per-feature drift and `drift_flag`. If `labels` supplied it also returns `continuous_accuracy`.

Example curl (predict):
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[14.2, 24.3, 10.1, 3.4, 0.2, 1.1, 0.0]]}'
```
(Feature vector length must match model's feature count — see "Input shape & feature names".)

Example curl (monitor):
```bash
curl -X POST http://127.0.0.1:5000/monitor \
  -H "Content-Type: application/json" \
  -d '{"features": [[14.2,24.3,10.1,3.4,0.2,1.1,0.0]], "labels": [1]}'
```

---

## Input shape & feature names
To get the expected feature names and order (so you can craft requests correctly):
```python
from src.data_preprocessing import get_feature_names
print(get_feature_names())
```
This reads the dataset CSV (downloaded by the data loader) and returns the feature columns (excluding the `Class` label).

---

## Docker — build & run

The repository includes a Dockerfile that:
- Uses `python:3.12-slim`
- Copies `src/`, `api/`, `models/` into `/app`
- Installs `numpy` and `flask`
- Exposes port 5000
- Runs `python api/app.py` as the container command

Build image:
```bash
docker build -t raisin-classifier .
```

Run container (map container port 5000 → host 5000):
```bash
docker run -p 5000:5000 raisin-classifier
```

Recommended for development: mount `models/` and `mlruns/` from host so the container can use trained artifacts you created locally:
```bash
docker run -p 5000:5000 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/mlruns:/app/mlruns" \
  raisin-classifier
```

Important: ensure the Flask server inside the container binds to 0.0.0.0 (see next section). Also note the Dockerfile copies `models/` at build time; if you build before training, the image may not include model artifacts — mounting or training inside the container solves that.

---

## Make the Flask server reachable from host (inside container)

The default dev server `app.run(debug=True)` binds to `127.0.0.1` which is only accessible inside the container. Update `api/app.py` to bind to all interfaces:

```python name=api/app.py url=https://github.com/MohammedHRashid/logistic_regression_from_scratch/blob/b9abaa7f23be54932d5fc0d720510d2525831c28/api/app.py#L74-L78
if __name__ == "__main__":
    # bind to 0.0.0.0 so the Flask dev server is reachable from the host when running in Docker
    app.run(host="0.0.0.0", debug=True)
```

After this change, rebuild the Docker image and run with `-p 5000:5000` and you should be able to access `http://localhost:5000/` on the host.

Production suggestion: use a WSGI server (gunicorn) and expose `0.0.0.0`:
```dockerfile
# example CMD for Dockerfile
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "api.app:app"]
```
and add `gunicorn` to dependencies.

---

## Docker permission denied: /var/run/docker.sock (your error)

If you see:
```
ERROR: permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: ...
```
This means your user cannot talk to the Docker daemon. Possible fixes:

Quick check (requires sudo):
```bash
sudo docker run --rm hello-world
```
If that works, the daemon is running but your user lacks access.

Linux: add user to `docker` group (recommended for convenience):
```bash
# add user to docker group
sudo usermod -aG docker $USER

# apply change to current session
newgrp docker

# test
docker run --rm hello-world
```
Log out and log back in to make group membership persistent.

If you don't want to add to docker group, prefix docker commands with `sudo`:
```bash
sudo docker build -t raisin-classifier .
sudo docker run -p 5000:5000 raisin-classifier
```

WSL2 on Windows:
- Ensure Docker Desktop is running on Windows.
- In Docker Desktop Settings → Resources → WSL Integration, enable your WSL distro.
- Restart Docker Desktop and your WSL shell.
- Then run `docker run hello-world` from WSL.

If you need exact step-by-step help for your environment (native Linux vs WSL2), tell me which and I will give tailored commands.

---

## Predict from Python (no API)

You can load saved model artifacts and predict from a script:

```python
# example_predict.py
import numpy as np
from src.predict import make_prediction

X = np.array([[14.2, 24.3, 10.1, 3.4, 0.2, 1.1, 0.0]])
preds, probs = make_prediction(X, model_path="models/<run-id>")
print("preds:", preds)
print("probs:", probs)
```

Run:
```bash
python example_predict.py
```

---

## Where artifacts are stored

- Runs and logs: `mlruns/<run-name>/` (created by `start_run`). Contains `params.json`, `metrics.json`, `baseline.json` and copies of `weights.npy`, `bias.npy` (if `log_model` was called).
- Models: `models/<run-name>/weights.npy` and `bias.npy` (created by `src/main.py`).

The API uses `get_latest_run()` to find the most recent run and load `weights.npy` and `bias.npy` from that run.

---

## Monitoring & explainability notes

- Data drift: `src/monitoring.compute_data_drift()` compares incoming data mean/std to the training baseline (`baseline.json`) and computes per-feature drift.
- Continuous evaluation: `src/monitoring.continuous_eval()` computes accuracy on new labeled batches and logs `continuous_accuracy`.
- Explainability: logistic regression weights are directly interpretable. Print `weights.npy` alongside `get_feature_names()` to inspect feature contributions.

Example:
```python
import numpy as np
from src.data_preprocessing import get_feature_names

weights = np.load("models/<run-id>/weights.npy")
names = get_feature_names()
for name, w in zip(names, weights):
    print(name, ":", w)
```
