"""Record from the microphone and run the int8 anger-detection model."""

import argparse
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import tensorflow as tf

from anger_detection.common import CLASS_NAMES, MODELS_DIR, SR, TIME_FRAMES
from anger_detection.features import log_mel_full


def record(seconds: float, sr: int = SR):
    print(f"Recording {seconds:.1f}s ... please speak now", flush=True)
    data = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Recording done.", flush=True)
    return data[:, 0]


def save_wav(path: Path, data: np.ndarray, sr: int = SR):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes())


def infer_int8(feat: np.ndarray, model_path: Path):
    x = feat.reshape(1, TIME_FRAMES, 40, 1).astype(np.float32)
    interp = tf.lite.Interpreter(model_path=str(model_path))
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
    return out[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0, help="recording length in seconds")
    parser.add_argument("--save", default=None, help="optional wav path to keep the recording")
    args = parser.parse_args()

    model_path = MODELS_DIR / "anger_detection_model_int8.tflite"
    if not model_path.exists():
        sys.exit("int8 tflite model not found - run export_tflite_torch.py first")

    audio = record(args.seconds)
    wav_path = Path(args.save) if args.save else Path(tempfile.gettempdir()) / "anger_test.wav"
    save_wav(wav_path, audio)
    print(f"audio saved to: {wav_path}")

    feat = log_mel_full(wav_path)
    t = feat.shape[0]
    if t >= TIME_FRAMES:
        start = (t - TIME_FRAMES) // 2
        patch = feat[start : start + TIME_FRAMES]
    else:
        pad_before = (TIME_FRAMES - t) // 2
        patch = np.pad(feat, ((pad_before, TIME_FRAMES - t - pad_before), (0, 0)), mode="edge")

    probs = infer_int8(patch, model_path)
    cls = int(np.argmax(probs))
    print(f"\npredicted: class {cls} -> {CLASS_NAMES[cls]}")
    for i, p in enumerate(probs):
        print(f"  {CLASS_NAMES[i]}: {p:.1%}")


if __name__ == "__main__":
    main()
