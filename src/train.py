"""
Training utilities for logistic regression.

Includes functions to:
- Train logistic regression with gradient descent
- Log training and v
"""
import numpy as np
from src.model import predict_proba, compute_loss
from src.mlflow_lite import log_metric


def train_logistic_regression(
    X,
    y,
    run_id=None,
    lr=0.01,
    epochs=1000,
    lambda_=0.0,
    X_val=None,
    y_val=None,
):
    """
    Train logistic regression with gradient descent.
    Logs training and validation metrics if run_id is provided.
    """
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    losses = []

    for i in range(epochs):
        # Forward pass
        y_hat = predict_proba(X, weights, bias)

        # Gradients
        dw = np.dot(X.T, (y_hat - y)) / m
        db = np.sum(y_hat - y) / m

        # L2 regularization
        if lambda_ > 0:
            dw += lambda_ * weights / m

        # Parameter update
        weights -= lr * dw
        bias -= lr * db

        # Training loss
        loss = compute_loss(
            y,
            y_hat,
            lambda_,
            weights if lambda_ > 0 else None,
        )
        losses.append(loss)

        # Log training metrics
        if run_id:
            log_metric(run_id, "train_loss", loss, step=i)

        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_hat = predict_proba(X_val, weights, bias)
            val_loss = compute_loss(y_val, y_val_hat)
            val_acc = np.mean((y_val_hat >= 0.5).astype(int) == y_val)
            if run_id:
                log_metric(run_id, "val_loss", val_loss, step=i)
                log_metric(run_id, "val_accuracy", val_acc, step=i)

        # Print progress every 100 epochs
        if i % 100 == 0 or i == epochs - 1:
            val_str = (
                f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                if X_val is not None else ""
            )
            print(f"Epoch {i}/{epochs}, Loss: {loss:.4f}{val_str}")

    return weights, bias, losses
