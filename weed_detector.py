from pathlib import Path
import threading

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    ort = None


MODEL_SIZE = 512
CLASS_NAMES = {0: "crop", 1: "weed"}
CLASS_NAMES_AR = {0: "محصول", 1: "عشبة"}


class WeedModelUnavailable(RuntimeError):
    pass


class WeedDetector:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._session = None
        self._input_name = None
        self._lock = threading.Lock()

    @property
    def available(self):
        return ort is not None and self.model_path.exists()

    def _load_session(self):
        if self._session is not None:
            return self._session

        if ort is None:
            raise WeedModelUnavailable("ONNX Runtime is not installed")
        if not self.model_path.exists():
            raise WeedModelUnavailable("Weed detection model is missing")

        with self._lock:
            if self._session is None:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
        return self._session

    @staticmethod
    def _prepare_image(image):
        original_width, original_height = image.size
        scale = min(MODEL_SIZE / original_width, MODEL_SIZE / original_height)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        left = (MODEL_SIZE - resized_width) // 2
        top = (MODEL_SIZE - resized_height) // 2

        resized = image.resize(
            (resized_width, resized_height),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (MODEL_SIZE, MODEL_SIZE), (114, 114, 114))
        canvas.paste(resized, (left, top))

        image_array = np.asarray(canvas, dtype=np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        tensor = np.expand_dims(image_array, axis=0)
        return np.ascontiguousarray(tensor), scale, left, top

    def predict(self, image, confidence_threshold=0.25):
        session = self._load_session()
        tensor, scale, left, top = self._prepare_image(image)
        output = session.run(None, {self._input_name: tensor})[0]
        rows = np.asarray(output)

        if rows.ndim == 3:
            rows = rows[0]
        if rows.ndim != 2 or rows.shape[1] < 6:
            raise RuntimeError(f"Unexpected ONNX output shape: {rows.shape}")

        original_width, original_height = image.size
        detections = []

        for row in rows:
            values = row[:6]
            if not np.isfinite(values).all():
                continue

            x1, y1, x2, y2, confidence, class_value = values
            confidence = float(confidence)
            class_id = int(round(float(class_value)))

            if confidence < confidence_threshold or class_id not in CLASS_NAMES:
                continue

            x1 = max(0.0, min(original_width, (float(x1) - left) / scale))
            y1 = max(0.0, min(original_height, (float(y1) - top) / scale))
            x2 = max(0.0, min(original_width, (float(x2) - left) / scale))
            y2 = max(0.0, min(original_height, (float(y2) - top) / scale))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "class_name_ar": CLASS_NAMES_AR[class_id],
                    "confidence": round(confidence, 4),
                    "box": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return {
            "image_width": original_width,
            "image_height": original_height,
            "detections": detections,
            "crop_count": sum(item["class_id"] == 0 for item in detections),
            "weed_count": sum(item["class_id"] == 1 for item in detections),
        }
