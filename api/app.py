from flask import Flask, request, jsonify
import numpy as np
from src.mlflow_lite import get_latest_run, log_metric
from src.model import predict
from src.monitoring import compute_data_drift, is_drift, continuous_eval

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Raisin classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json  # expects {"features": [[...], [...]]}
    features = np.array(data["features"], dtype=float)
    
    latest = get_latest_run()
    if latest is None:
        return jsonify({"error": "No model available"}), 400
    model_dir = latest["run_dir"]
    
    weights = np.load(f"{model_dir}/weights.npy")
    bias = np.load(f"{model_dir}/bias.npy")
    
    probs = 1 / (1 + np.exp(-np.dot(features, weights) - bias))
    preds = (probs >= 0.5).astype(int)
    
    return jsonify({
        "predictions": preds.tolist(),
        "probabilities": probs.tolist()
    })

@app.route("/monitor", methods=["POST"])
def monitor_api():
    data = request.json
    X_new = np.array(data["features"], dtype=float)
    y_new = np.array(data.get("labels", []), dtype=int) if "labels" in data else None

    latest = get_latest_run()
    if latest is None:
        return jsonify({"error": "No model found"}), 400
    run_dir = latest["run_dir"]

    # ---- Data Drift ----
    try:
        drift = compute_data_drift(X_new, run_dir)
        drift_flag = is_drift(drift)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Log drift metrics
    for i, d in enumerate(drift):
        log_metric(run_dir, f"drift_feature_{i}", float(d))
    log_metric(run_dir, "drift_flag", float(drift_flag))

    response = {
        "drift_per_feature": drift.tolist(),
        "drift_flag": bool(drift_flag)
    }

    # ---- Continuous Evaluation ----
    if y_new is not None and y_new.size > 0:
        acc = continuous_eval(X_new, y_new, run_dir=run_dir)
        response["continuous_accuracy"] = float(acc)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
