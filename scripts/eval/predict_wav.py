"""Run a single wav through the int8 TFLite model (PC-side sanity check)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from anger_detection.common import CLASS_NAMES, MODELS_DIR, TIME_FRAMES
from anger_detection.features import log_mel_full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="path to a wav file")
    args = parser.parse_args()
    path = MODELS_DIR / "anger_detection_model_int8.tflite"
    if not path.exists():
        sys.exit("int8 tflite model not found - run quantize.py first")

    feat = log_mel_full(Path(args.wav))  # (T, 40)
    t = feat.shape[0]
    if t >= TIME_FRAMES:
        start = (t - TIME_FRAMES) // 2
        patch = feat[start : start + TIME_FRAMES]
    else:
        pad_before = (TIME_FRAMES - t) // 2
        patch = np.pad(feat, ((pad_before, TIME_FRAMES - t - pad_before), (0, 0)), mode="edge")
    x = patch.reshape(1, TIME_FRAMES, 40, 1).astype(np.float32)

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    if in_det["dtype"] == np.int8:
        scale, zp = in_det["quantization"]
        x = (x / scale + zp).round().astype(np.int8)
    interp.set_tensor(in_det["index"], x)
    interp.invoke()
    out = interp.get_tensor(out_det["index"])
    if out_det["dtype"] == np.int8:
        scale, zp = out_det["quantization"]
        out = (out.astype(np.float32) - zp) * scale
    probs = out[0]
    cls = int(np.argmax(probs))
    print(f"file: {args.wav}")
    print(f"predicted class {cls}: {CLASS_NAMES[cls]}")
    print("scores:", {CLASS_NAMES[i].split('_')[0]: round(float(p), 4) for i, p in enumerate(probs)})


if __name__ == "__main__":
    main()
