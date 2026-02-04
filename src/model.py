import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_proba(X, weights, bias):
    """Return probabilities for logistic regression"""
    return sigmoid(np.dot(X, weights) + bias)


def predict(X, weights, bias):
    probs = predict_proba(X, weights, bias)
    return (probs >= 0.5).astype(int), probs


def compute_loss(y, y_hat, lambda_=0.0, weights=None):
    """Binary cross-entropy with optional L2 regularization"""
    m = len(y)
    eps = 1e-15  # avoid log(0)
    y_hat = np.clip(y_hat, eps, 1 - eps)
    loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    if lambda_ > 0 and weights is not None:
        loss += (lambda_ / 2) * np.sum(weights**2)
    return loss
