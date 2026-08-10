from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.csv"
MODEL_FILE = BASE_DIR / "real_data_nn_model.pkl"
SCALER_FILE = BASE_DIR / "real_data_scaler.pkl"
METRICS_FILE = BASE_DIR / "model_metrics.json"
CLEAN_DATA_FILE = BASE_DIR / "cleaned_data.csv"
PREDICTIONS_FILE = BASE_DIR / "test_predictions.csv"

FEATURES = ["temperature", "pressure", "soil_moisture"]
CLASS_ORDER = ["Very Dry", "Dry", "Wet", "Very Wet"]


def load_and_clean_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)

    required = {"temperature", "pressure", "soilmiosture", "class"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.rename(columns={"soilmiosture": "soil_moisture"})
    before_count = len(data)

    valid_rows = (
        data["temperature"].between(0, 60)
        & data["pressure"].between(8000, 11000)
        & data["soil_moisture"].between(0, 500)
        & data["class"].isin(CLASS_ORDER)
    )

    cleaned = data.loc[valid_rows].copy()
    removed_count = before_count - len(cleaned)
    cleaned.to_csv(CLEAN_DATA_FILE, index=False)

    return cleaned, before_count, removed_count


def train_model():
    data, original_count, removed_count = load_and_clean_data()

    x = data[FEATURES]
    y = data["class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )
    model.fit(x_train_scaled, y_train)

    predictions = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=CLASS_ORDER)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    result_rows = x_test.reset_index(drop=True).copy()
    result_rows["actual_class"] = y_test.reset_index(drop=True)
    result_rows["predicted_class"] = predictions
    result_rows.to_csv(PREDICTIONS_FILE, index=False)

    metrics = {
        "dataset": {
            "original_rows": int(original_count),
            "clean_rows": int(len(data)),
            "removed_invalid_rows": int(removed_count),
            "training_rows": int(len(x_train)),
            "testing_rows": int(len(x_test)),
        },
        "model": {
            "name": "MLPClassifier",
            "hidden_layers": [64, 32],
            "iterations": int(model.n_iter_),
            "features": FEATURES,
        },
        "performance": {
            "accuracy": float(accuracy),
            "weighted_precision": float(precision),
            "weighted_recall": float(recall),
            "weighted_f1_score": float(f1),
        },
        "class_order": CLASS_ORDER,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print("Smart Agriculture Gaza AI")
    print("=" * 42)
    print(f"Original rows: {original_count}")
    print(f"Removed invalid rows: {removed_count}")
    print(f"Clean rows: {len(data)}")
    print(f"Training rows: {len(x_train)}")
    print(f"Testing rows: {len(x_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    print(f"Training iterations: {model.n_iter_}")
    print(f"Model saved: {MODEL_FILE.name}")
    print(f"Scaler saved: {SCALER_FILE.name}")


if __name__ == "__main__":
    train_model()
