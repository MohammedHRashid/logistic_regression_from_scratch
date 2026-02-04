import json
import os
from datetime import datetime

import numpy as np


def start_run(run_name=None):
    """Start a new run directory and return its path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = run_name or f"run_{timestamp}"
    run_dir = os.path.join("mlruns", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def log_params(run_dir, params):
    """Log parameters to a JSON file."""
    path = os.path.join(run_dir, "params.json")
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(params, file_handle, indent=2)


def log_metrics(run_dir, metrics):
    """Log metrics to a JSON file."""
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)


def log_model(run_dir, model_dir):
    """
    Save model artifacts (weights and bias) from model_dir into run_dir.
    """
    for file_name in ("weights.npy", "bias.npy"):
        src_path = os.path.join(model_dir, file_name)
        dst_path = os.path.join(run_dir, file_name)
        if os.path.exists(src_path):
            np.save(dst_path, np.load(src_path))


def log_metric(run_dir, name, value, step=None):
    """
    Log a single metric value at a specific step.
    """
    path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file_handle:
            metrics = json.load(file_handle)
    else:
        metrics = {}

    metrics.setdefault(name, [])
    metrics[name].append({"step": step, "value": value})

    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)


def log_baseline(run_dir, x_train):
    """Log baseline statistics (mean and std) of training data."""
    baseline = {
        "mean": x_train.mean(axis=0).tolist(),
        "std": x_train.std(axis=0).tolist(),
    }
    path = os.path.join(run_dir, "baseline.json")
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(baseline, file_handle, indent=2)


def load_baseline(run_dir):
    """Load baseline statistics if they exist, else return None."""
    path = os.path.join(run_dir, "baseline.json")
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def get_latest_run():
    """Fetch the latest run directory and its parameters."""
    if not os.path.exists("mlruns"):
        return None

    runs = sorted(os.listdir("mlruns"))
    if not runs:
        return None

    latest_run = runs[-1]
    run_dir = os.path.join("mlruns", latest_run)
    params_path = os.path.join(run_dir, "params.json")

    if not os.path.exists(params_path):
        return {"run_dir": run_dir, "params": {}}

    with open(params_path, "r", encoding="utf-8") as file_handle:
        params = json.load(file_handle)

    return {"run_dir": run_dir, "params": params}