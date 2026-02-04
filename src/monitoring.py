import numpy as np
from src.mlflow_lite import load_baseline, get_latest_run, log_metric
from src.model import predict


def compute_data_drift(X_new, run_dir):
    """
    Compare new incoming data X_new with baseline stats.
    Returns per-feature drift score (difference in means normalized by baseline std)
    """
    baseline = load_baseline(run_dir)
    if baseline is None:
        raise ValueError("No baseline found. Cannot compute drift.")

    baseline_mean = np.array(baseline["mean"])
    baseline_std = np.array(baseline["std"])

    # Avoid division by zero
    baseline_std = np.where(baseline_std == 0, 1e-6, baseline_std)

    mean_new = X_new.mean(axis=0)
    drift = np.abs(mean_new - baseline_mean) / baseline_std

    # Return drift per feature
    return drift

def is_drift(drift, threshold=2.0):
    """
    Simple heuristic: if any feature drift > threshold, flag as drift
    """
    return np.any(drift > threshold)


def continuous_eval(X_new, y_new, run_dir=None, step=None):
    """
    Evaluate model accuracy on new data and log metric
    """
    if run_dir is None:
        latest = get_latest_run()
        if latest is None:
            raise ValueError("No model available for evaluation")
        run_dir = latest["run_dir"]

    # Load model
    weights = np.load(f"{run_dir}/weights.npy")
    bias = np.load(f"{run_dir}/bias.npy")

    # Predict
    y_pred, _ = predict(X_new, weights, bias)
    acc = np.mean(y_pred == y_new)

    # Log metric
    log_metric(run_dir, "continuous_accuracy", acc, step=step)
    return acc