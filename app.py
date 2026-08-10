from pathlib import Path
import math
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "real_data_nn_model.pkl"
SCALER_FILE = BASE_DIR / "real_data_scaler.pkl"
FEATURES = ["temperature", "pressure", "soil_moisture"]

CLASS_DETAILS = {
    "Very Dry": {
        "status_ar": "جافة جدًا",
        "action_ar": "تشغيل المضخة بالطاقة القصوى لمدة 15 دقيقة",
        "relay_state": 1,
        "css_class": "very-dry",
    },
    "Dry": {
        "status_ar": "جافة",
        "action_ar": "تشغيل المضخة بضخ عادي لمدة 8 دقائق",
        "relay_state": 1,
        "css_class": "dry",
    },
    "Wet": {
        "status_ar": "رطبة",
        "action_ar": "لا حاجة للري، إيقاف المضخة",
        "relay_state": 0,
        "css_class": "wet",
    },
    "Very Wet": {
        "status_ar": "رطبة جدًا",
        "action_ar": "التربة مشبعة، إيقاف المضخة فورًا",
        "relay_state": 0,
        "css_class": "very-wet",
    },
}


if not MODEL_FILE.exists() or not SCALER_FILE.exists():
    raise RuntimeError("Model files are missing. Run: python train_model.py")

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

app = Flask(__name__)
CORS(app)


def read_number(data, key):
    if key not in data:
        raise ValueError(f"Missing field: {key}")

    value = float(data[key])
    if not math.isfinite(value):
        raise ValueError(f"Invalid numeric value: {key}")
    return value


def predict_soil_state(temperature, pressure, soil_moisture):
    if not 0 <= temperature <= 60:
        raise ValueError("Temperature must be between 0 and 60")
    if not 8000 <= pressure <= 11000:
        raise ValueError("Pressure must be between 8000 and 11000")
    if not 0 <= soil_moisture <= 500:
        raise ValueError("Soil moisture must be between 0 and 500")

    input_data = pd.DataFrame(
        [[temperature, pressure, soil_moisture]],
        columns=FEATURES,
    )
    scaled_input = scaler.transform(input_data)
    predicted_class = str(model.predict(scaled_input)[0])

    if predicted_class not in CLASS_DETAILS:
        raise ValueError(f"Unknown model class: {predicted_class}")

    result = dict(CLASS_DETAILS[predicted_class])
    result["class"] = predicted_class
    return result


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "Smart Agriculture Gaza AI",
            "model": "MLPClassifier",
            "classes": list(CLASS_DETAILS.keys()),
        }
    )


@app.post("/predict_web")
def predict_web():
    try:
        data = request.get_json(silent=False)
        if not isinstance(data, dict):
            raise ValueError("The request body must be a JSON object")

        temperature = read_number(data, "temp")
        pressure = read_number(data, "pressure")
        moistures = data.get("moistures")

        if not isinstance(moistures, list) or not moistures:
            raise ValueError("moistures must be a non-empty list")

        results = []
        for zone_number, moisture_value in enumerate(moistures, start=1):
            soil_moisture = float(moisture_value)
            if not math.isfinite(soil_moisture):
                raise ValueError("Every soil moisture value must be numeric")

            result = predict_soil_state(temperature, pressure, soil_moisture)
            result["zone"] = zone_number
            result["soil_moisture"] = soil_moisture
            results.append(result)

        return jsonify({"success": True, "results": results})
    except (TypeError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 400


@app.post("/api/esp32")
def esp32_telemetry():
    try:
        data = request.get_json(silent=False)
        if not isinstance(data, dict):
            raise ValueError("The request body must be a JSON object")

        temperature = read_number(data, "temp")
        pressure = read_number(data, "pressure")
        soil_moisture = read_number(data, "moisture")
        result = predict_soil_state(temperature, pressure, soil_moisture)

        return jsonify(
            {
                "success": True,
                "soil_status": result["status_ar"],
                "class": result["class"],
                "relay_state": result["relay_state"],
            }
        )
    except (TypeError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
