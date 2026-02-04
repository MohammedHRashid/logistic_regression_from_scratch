import numpy as np
from src.mlflow_lite import load_baseline, get_latest_run, log_metric
from src.model import predict


def compute_data_drift(X_new, run_dir, log=False, step=None):
    """
    Compute per-feature data drift relative to baseline statistics.

    Args:
        X_new: New incoming data (samples x features).
        run_dir: Directory of the MLflow run containing baseline stats.
        log: If True, logs drift per feature using MLflow.
        step: Step number for logging.

    Returns:
        Array of per-feature drift scores.
    """
    baseline = load_baseline(run_dir)
    if baseline is None:
        raise ValueError("No baseline found. Cannot compute drift.")

    baseline_mean = np.array(baseline["mean"])
    baseline_std = np.where(np.array(baseline["std"]) == 0, 1e-6, baseline["std"])

    drift = np.abs(X_new.mean(axis=0) - baseline_mean) / baseline_std

    if log:
        for i, d in enumerate(drift):
            log_metric(run_dir, f"drift_feature_{i}", float(d), step=step)

    return drift


def is_drift(drift, threshold=2.0):
    """
    Determine if data drift exceeds a threshold.

    Args:
        drift: Array of per-feature drift scores.
        threshold: Threshold for flagging drift.

    Returns:
        True if any feature exceeds the threshold.
    """
    return np.any(drift > threshold)


def continuous_eval(X_new, y_new, run_dir=None, step=None):
    """
    Evaluate model accuracy on new data and log the metric.

    Args:
        X_new: New feature data.
        y_new: True labels for X_new.
        run_dir: Directory of the MLflow run. If None, uses latest run.
        step: Step number for logging.

    Returns:
        Accuracy on new data.
    """
    if run_dir is None:
        latest = get_latest_run()
        if latest is None:
            raise ValueError("No model available for evaluation.")
        run_dir = latest["run_dir"]

    # Load model
    weights = np.load(f"{run_dir}/weights.npy")
    bias = np.load(f"{run_dir}/bias.npy")

    # Predict
    y_pred, _ = predict(X_new, weights, bias)
    acc = float(np.mean(y_pred == y_new))

    # Log metric
    log_metric(run_dir, "continuous_accuracy", acc, step=step)
    return acc