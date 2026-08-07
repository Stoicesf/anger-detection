"""Export float32 and int8 TFLite models with PTQ calibration."""

import json

import numpy as np
import tensorflow as tf

from anger_detection.common import (
    CLASS_NAMES,
    MODELS_DIR,
    NUM_CLASSES,
    PROCESSED_DIR,
    ensure_dirs,
)


def representative_dataset(X_train, n=100):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_train), size=n, replace=False)
    for i in idx:
        yield [X_train[i].reshape(1, 98, 40, 1).astype(np.float32)]


def main() -> None:
    ensure_dirs()
    model = tf.keras.models.load_model(MODELS_DIR / "dscnn_final.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()
    (MODELS_DIR / "anger_detection_model_fp32.tflite").write_bytes(tflite_fp32)
    print("fp32 tflite:", len(tflite_fp32) / 1024, "KB")

    X = np.load(PROCESSED_DIR / "X.npy")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(X)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()
    (MODELS_DIR / "anger_detection_model_int8.tflite").write_bytes(tflite_int8)
    print("int8 tflite:", len(tflite_int8) / 1024, "KB")

    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump({"class_names": CLASS_NAMES, "num_classes": NUM_CLASSES}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
