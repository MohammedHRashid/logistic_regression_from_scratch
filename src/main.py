# main.py
import os
import numpy as np
import json
from src.data_preprocessing import load_data, train_test_split, get_feature_names
from src.train import train_logistic_regression
from src.model import predict
from src.mlflow_lite import start_run, log_model, log_baseline


def main():
    # ---------------- Hyperparameters ----------------
    LR = 0.01
    EPOCHS = 100
    LAMBDA = 0.01

    params = {"lr": LR, "epochs": EPOCHS, "lambda": LAMBDA}

    # ---------------- Start Experiment ----------------
    run_dir = start_run()
    log_params_path = os.path.join(run_dir, "params.json")
    # Save hyperparameters
    with open(log_params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Starting experiment {run_dir} with params: {params}")

    # ---------------- Load & Preprocess Data ----------------
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # ---------------- Split a small validation set from training (10%) ------
    val_size = int(0.1 * len(X_train))
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train2, y_train2 = X_train[val_size:], y_train[val_size:]

    # ---------------- Log baseline for monitoring ----------------
    log_baseline(run_dir, X_train2)
    print("Baseline statistics logged for drift detection.")

    # ---------------- Train Model ----------------
    weights, bias, _ = train_logistic_regression(
        X_train2,
        y_train2,
        run_id=run_dir,
        lr=LR,
        epochs=EPOCHS,
        lambda_=LAMBDA,
        X_val=X_val,
        y_val=y_val
    )

    # ---------------- Save Model ----------------
    model_version = os.path.basename(run_dir)
    model_dir = os.path.join("models", model_version)
    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(model_dir, "weights.npy"), weights)
    np.save(os.path.join(model_dir, "bias.npy"), bias)
    log_model(run_dir, model_dir)
    print(f"Model saved to {model_dir}")

    # ---------------- Evaluate on Test ----------------
    y_pred, _ = predict(X_test, weights, bias)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
