"""
Data preprocessing utilities for the Raisin dataset.

Includes functions to:
- Download the dataset from Kaggle if missing
- Load and standardize features
- Encode binary labels
- Split data into training and test sets
- Get feature column names
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np


def download_dataset():
    """Download Raisin dataset if missing."""
    dataset_dir = Path("data/raw")
    dataset_file = dataset_dir / "Raisin_Dataset.csv"

    if not dataset_file.exists():
        print("Dataset not found. Downloading from Kaggle...")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        os.system(
            f"kaggle datasets download -d nimapourmoradi/raisin-binary-classification "
            f"-p {dataset_dir} --unzip"
        )
        if not dataset_file.exists():
            raise FileNotFoundError(
                "Dataset download failed. Make sure Kaggle API is set up correctly."
            )
        print("Dataset downloaded successfully.")
    return dataset_file


def load_data():
    """Load and preprocess Raisin dataset (standardized, labels binary)."""
    dataset_file = download_dataset()
    data = pd.read_csv(dataset_file)

    # Binary encode labels (Kecimen=1, Besni=0)
    y = np.where(data["Class"] == "Kecimen", 1, 0)

    # Extract features
    X = data.drop("Class", axis=1).values.astype(float)

    # Standardize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X, y


def train_test_split(X, y, test_size=0.2, random_seed=42):
    """
    Split dataset into train and test sets.

    Args:
        X: features np.ndarray
        y: labels np.ndarray
        test_size: fraction of data for test
        random_seed: random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    X, y = X[idxs], y[idxs]

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


def get_feature_names():
    """Return list of feature column names."""
    dataset_file = download_dataset()
    data = pd.read_csv(dataset_file)
    return [col for col in data.columns if col != "Class"]
