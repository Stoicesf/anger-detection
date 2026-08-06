"""Convert the trained ONNX model to int8 TFLite via onnx2tf, then validate."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import MODELS_DIR, PROCESSED_DIR, ensure_dirs


def main() -> None:
    ensure_dirs()
    onnx_path = MODELS_DIR / "dscnn.onnx"
    if not onnx_path.exists():
        sys.exit("dscnn.onnx not found - run train_torch.py first")

    X = np.load(PROCESSED_DIR / "X.npy")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=200, replace=False)
    calib = X[idx].astype(np.float32)  # (200, 98, 40, 1) already in [0,1]
    calib_path = MODELS_DIR / "calib_data.npy"
    np.save(calib_path, calib)

    out_dir = MODELS_DIR / "tflite_export"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(out_dir),
        "-oiqt",
        "-iqd", "int8",
        "-cind", "input", str(calib_path), "0", "1",
        "-tb", "tf_converter",
        "-b", "1",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    produced = sorted(out_dir.glob("*.tflite"))
    if not produced:
        sys.exit("onnx2tf produced no tflite")
    final = MODELS_DIR / "anger_detection_model_int8.tflite"
    shutil.copyfile(produced[0], final)
    print("int8 tflite ->", final, final.stat().st_size / 1024, "KB")

    # also make a float32 tflite (backend tf_converter) for comparison
    out_fp32 = MODELS_DIR / "tflite_fp32"
    if out_fp32.exists():
        shutil.rmtree(out_fp32)
    out_fp32.mkdir(parents=True)
    cmd_fp32 = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(out_fp32),
        "-tb", "tf_converter",
        "-b", "1",
    ]
    subprocess.run(cmd_fp32, check=True)
    produced_fp32 = sorted(out_fp32.glob("*.tflite"))
    if produced_fp32:
        fp32_final = MODELS_DIR / "anger_detection_model_fp32.tflite"
        shutil.copyfile(produced_fp32[0], fp32_final)
        print("fp32 tflite ->", fp32_final, fp32_final.stat().st_size / 1024, "KB")

    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        from common import CLASS_NAMES, NUM_CLASSES

        json.dump({"class_names": CLASS_NAMES, "num_classes": NUM_CLASSES}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
