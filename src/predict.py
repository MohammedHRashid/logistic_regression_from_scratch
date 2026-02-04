import numpy as np
from src.model import predict


def load_model(model_path="models/v1"):
    weights = np.load(f"{model_path}/weights.npy")
    bias = np.load(f"{model_path}/bias.npy")
    return weights, bias


def make_prediction(X, model_path="models/v1"):
    weights, bias = load_model(model_path)
    return predict(X, weights, bias)
