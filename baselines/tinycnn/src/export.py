"""
Export TinyCNN to ONNX (+ optional calibration .npy for INT8).

Usage:
  python src/export.py
  python src/export.py --make_calib
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import AngryDataset
from feature import extract_dual_channel, extract_mfcc
from model import TinyCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "best_model.pth"))
    parser.add_argument("--out_dir", type=str, default=str(ROOT / "exported"))
    parser.add_argument("--time_frames", type=int, default=201, help="2s @ hop=160 -> ~201")
    parser.add_argument("--make_calib", action="store_true", help="dump calibration_data.npy")
    parser.add_argument("--calib_n", type=int, default=200)
    parser.add_argument("--data_root", type=str, default=str(ROOT / "dataset"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", 1)
    dual = ckpt.get("dual", in_channels == 2)

    model = TinyCNN(in_channels=in_channels)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # wrap with sigmoid for deployment-friendly probability output
    class ExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.sigmoid(self.m(x))

    export_model = ExportWrapper(model)

    T = args.time_frames
    dummy = torch.randn(1, in_channels, 40, T)
    onnx_path = out_dir / "model.onnx"

    torch.onnx.export(
        export_model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch_size", 3: "time_frames"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX exported: {onnx_path}")

    # quick ORT check if available
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": dummy.numpy()})[0]
        print(f"ORT smoke output shape={out.shape}, value={out.flatten()[:4]}")
    except Exception as e:
        print(f"[Warn] onnxruntime check skipped: {e}")

    if args.make_calib:
        feature_fn = extract_dual_channel if dual else extract_mfcc
        val_ds = AngryDataset(
            os.path.join(args.data_root, "val"),
            feature_fn=feature_fn,
            augment=False,
        )
        loader = DataLoader(val_ds, batch_size=1, shuffle=True)
        feats = []
        for i, (x, _) in enumerate(loader):
            if i >= args.calib_n:
                break
            feats.append(x.numpy())
        calib = np.concatenate(feats, axis=0)
        calib_path = out_dir / "calibration_data.npy"
        np.save(calib_path, calib)
        print(f"Calibration data: {calib_path} shape={calib.shape}")

    print(
        "\nNext (optional INT8):\n"
        "  pip install onnx2tf\n"
        "  onnx2tf -i exported/model.onnx -o exported/tflite_fp32\n"
        "  onnx2tf -i exported/model.onnx -o exported/tflite_int8 -qt INT8 "
        "-c exported/calibration_data.npy"
    )


if __name__ == "__main__":
    main()
