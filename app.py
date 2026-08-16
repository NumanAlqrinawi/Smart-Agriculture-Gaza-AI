from pathlib import Path
from io import BytesIO
import math
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

from weed_detector import CLASS_NAMES, WeedDetector, WeedModelUnavailable


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "real_data_nn_model.pkl"
SCALER_FILE = BASE_DIR / "real_data_scaler.pkl"
WEED_MODEL_FILE = BASE_DIR / "weed_detector.onnx"
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
weed_detector = WeedDetector(WEED_MODEL_FILE)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024
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
            "weed_detector": {
                "model": "YOLO11n ONNX",
                "available": weed_detector.available,
                "classes": list(CLASS_NAMES.values()),
            },
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


@app.post("/predict_weed")
def predict_weed():
    try:
        image_file = request.files.get("image")
        if image_file is None or not image_file.filename:
            raise ValueError("Please select an image")

        confidence = float(request.form.get("confidence", "0.35"))
        if not 0.1 <= confidence <= 0.9:
            raise ValueError("Confidence must be between 0.1 and 0.9")

        image_bytes = image_file.read()
        if not image_bytes:
            raise ValueError("The uploaded image is empty")

        image = Image.open(BytesIO(image_bytes))
        if image.width * image.height > 16_000_000:
            raise ValueError("Image resolution is too large")
        image.load()
        image = image.convert("RGB")

        result = weed_detector.predict(image, confidence)
        result["success"] = True
        result["model"] = "YOLO11n"
        return jsonify(result)
    except WeedModelUnavailable as error:
        return jsonify({"success": False, "error": str(error)}), 503
    except (
        TypeError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
    ) as error:
        return jsonify({"success": False, "error": str(error)}), 400
    except Exception:
        app.logger.exception("Weed image analysis failed")
        return jsonify(
            {"success": False, "error": "The image could not be analyzed"}
        ), 500


@app.errorhandler(413)
def image_too_large(_error):
    return jsonify({"success": False, "error": "Image must be smaller than 6 MB"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
