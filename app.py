
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "artifacts" / "crop_model.joblib"

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Crop Recommendation API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing": missing
        }), 400

    try:
        values = [float(data[feature]) for feature in FEATURES]
    except (TypeError, ValueError):
        return jsonify({
            "error": "All features must be numeric"
        }), 400

    if any(value < 0 for value in values):
        return jsonify({
            "error": "Features cannot be negative"
        }), 400

    if not MODEL_PATH.exists():
        return jsonify({
            "error": "Trained model not found"
        }), 500

    model = joblib.load(MODEL_PATH)
    input_data = pd.DataFrame([values], columns=FEATURES)
    predicted_crop = model.predict(input_data)[0]

    return jsonify({
        "predicted_crop": str(predicted_crop)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
