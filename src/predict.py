import numpy as np
from src.model import predict


def load_model(model_path="models/v1"):
    """
    Load model weights and bias from the specified path.
    """
    weights = np.load(f"{model_path}/weights.npy")
    bias = np.load(f"{model_path}/bias.npy")
    return weights, bias


def make_prediction(X, model_path="models/v1"):
    """
    Make predictions using the logistic regression model.
    
    Args:
        X (np.ndarray): Input features.
        model_path (str): Path to the model directory.

    Returns:
        Tuple of (predictions, probabilities)
    """
    weights, bias = load_model(model_path)
    return predict(X, weights, bias)