"""
Generate a full performance report JSON for the best checkpoint.

Usage:
  python src/report.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import AngryDataset
from feature import extract_dual_channel, extract_mfcc
from model import TinyCNN, count_parameters


@torch.no_grad()
def eval_split(model, loader, device, threshold=0.5):
    ys, ps, probs_all = [], [], []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features).squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        ys.append(labels.cpu().numpy())
        ps.append(preds.cpu().numpy())
        probs_all.append(probs.cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    probs = np.concatenate(probs_all)

    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    total = len(y)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    acc = (tp + tn) / max(total, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "n": total,
        "n_angry": int((y == 1).sum()),
        "n_non_angry": int((y == 0).sum()),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "prob_mean_angry": float(probs[y == 1].mean()) if (y == 1).any() else None,
        "prob_mean_non_angry": float(probs[y == 0].mean()) if (y == 0).any() else None,
    }


@torch.no_grad()
def measure_latency(model, sample, device, warmup=20, runs=100):
    x = sample.unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(ROOT / "dataset"))
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "best_model.pth"))
    parser.add_argument("--out", type=str, default=str(ROOT / "checkpoints" / "performance_report.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", 1)
    dual = ckpt.get("dual", in_channels == 2)
    feature_fn = extract_dual_channel if dual else extract_mfcc

    model = TinyCNN(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    splits = {}
    sample_feat = None
    for name in ("train", "val", "test"):
        ds = AngryDataset(
            os.path.join(args.data_root, name),
            feature_fn=feature_fn,
            augment=False,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        splits[name] = eval_split(model, loader, device, threshold=args.threshold)
        if sample_feat is None and len(ds) > 0:
            sample_feat, _ = ds[0]

    latency = measure_latency(model, sample_feat, device) if sample_feat is not None else None
    ckpt_size = os.path.getsize(args.ckpt)

    history = []
    hist_path = ROOT / "checkpoints" / "train_history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text(encoding="utf-8"))

    summary = {}
    sum_path = ROOT / "checkpoints" / "train_summary.json"
    if sum_path.exists():
        summary = json.loads(sum_path.read_text(encoding="utf-8"))

    report = {
        "model": {
            "name": "TinyCNN",
            "params": count_parameters(model),
            "in_channels": in_channels,
            "dual": dual,
            "ckpt": args.ckpt,
            "ckpt_bytes": ckpt_size,
            "best_epoch": ckpt.get("epoch"),
            "recorded_val_acc": ckpt.get("val_acc"),
            "recorded_val_f1": ckpt.get("val_f1"),
            "augment": ckpt.get("augment"),
        },
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "threshold": args.threshold,
        "splits": splits,
        "latency": latency,
        "train_summary": summary,
        "history": history,
    }

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model", "hardware", "splits", "latency")}, indent=2))
    print(f"\nSaved report -> {out}")


if __name__ == "__main__":
    main()
