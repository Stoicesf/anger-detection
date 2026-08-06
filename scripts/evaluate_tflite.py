"""Validate the exported TFLite models against the held-out test split."""

import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from common import CLASS_NAMES, MODELS_DIR, OUTPUT_DIR, PROCESSED_DIR


def eval_tflite(path, X, y):
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    preds = []
    for i in range(len(X)):
        x = X[i : i + 1].astype(np.float32)
        if in_det["dtype"] == np.int8:
            scale, zp = in_det["quantization"]
            x = (x / scale + zp).round().astype(np.int8)
        interp.set_tensor(in_det["index"], x)
        interp.invoke()
        out = interp.get_tensor(out_det["index"])
        if out_det["dtype"] == np.int8:
            scale, zp = out_det["quantization"]
            out = (out.astype(np.float32) - zp) * scale
        preds.append(out.argmax())
    return np.array(preds)


def main() -> None:
    X = np.load(PROCESSED_DIR / "X_test.npy")
    y = np.load(PROCESSED_DIR / "y_test.npy")
    print(f"test set: {X.shape}, {len(y)} samples")

    results = {}
    for name in ("anger_detection_model_fp32.tflite", "anger_detection_model_int8.tflite"):
        path = MODELS_DIR / name
        if not path.exists():
            continue
        pred = eval_tflite(path, X, y)
        acc = float((pred == y).mean())
        results[name] = {"accuracy": acc}
        print(f"\n=== {name}  accuracy={acc:.4f} ===")
        print(classification_report(y, pred, labels=list(range(5)), target_names=CLASS_NAMES, zero_division=0))
        print("confusion matrix:\n", confusion_matrix(y, pred, labels=list(range(5))))

    with open(OUTPUT_DIR / "tflite_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
