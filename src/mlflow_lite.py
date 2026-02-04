# src/mlflow_lite.py
import json
import os
import numpy as np
from datetime import datetime


def start_run(run_name=None):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = run_name or f"run_{ts}"
    run_dir = os.path.join("mlruns", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def log_params(run_dir, params: dict):
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)


def log_metrics(run_dir, metrics: dict):
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def log_model(run_dir, model_dir):
    """
    Save model artifacts (weights/bias) from model_dir into run_dir.
    """
    for file_name in ["weights.npy", "bias.npy"]:
        src_path = os.path.join(model_dir, file_name)
        dst_path = os.path.join(run_dir, file_name)
        if os.path.exists(src_path):
            np.save(dst_path, np.load(src_path))

# -----------------------------
# Log a single metric (for train/val)
# -----------------------------


def log_metric(run_dir, name, value, step=None):
    path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics[name] = metrics.get(name, [])
    metrics[name].append({"step": step, "value": value})
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

# -----------------------------
# Baseline stats
# -----------------------------


def log_baseline(run_dir, X_train):
    baseline = {
        "mean": X_train.mean(axis=0).tolist(),
        "std": X_train.std(axis=0).tolist()
    }
    with open(os.path.join(run_dir, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)


def load_baseline(run_dir):
    path = os.path.join(run_dir, "baseline.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# -----------------------------
# Fetch latest run
# -----------------------------


def get_latest_run():
    runs = sorted(os.listdir("mlruns"))
    if not runs:
        return None
    latest = runs[-1]
    run_dir = os.path.join("mlruns", latest)
    with open(os.path.join(run_dir, "params.json")) as f:
        params = json.load(f)
    return {"run_dir": run_dir, "params": params}
