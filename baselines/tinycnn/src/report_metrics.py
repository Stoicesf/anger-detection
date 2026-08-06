"""
Generate detailed evaluation metrics + training curve parse for report.

Usage:
  python src/report_metrics.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import AngryDataset
from feature import extract_dual_channel, extract_mfcc
from model import TinyCNN, count_parameters


def parse_train_log(log_text: str) -> list[dict]:
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s*\|\s*"
        r"train_loss=([\d.]+)\s*\|\s*"
        r"val_loss=([\d.]+)\s*\|\s*"
        r"val_acc=([\d.]+)\s*\|\s*"
        r"val_f1=([\d.]+)\s*\|\s*"
        r"P=([\d.]+)\s*R=([\d.]+)"
    )
    rows = []
    for m in pattern.finditer(log_text):
        rows.append(
            {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(3)),
                "val_loss": float(m.group(4)),
                "val_acc": float(m.group(5)),
                "val_f1": float(m.group(6)),
                "precision": float(m.group(7)),
                "recall": float(m.group(8)),
            }
        )
    return rows


@torch.no_grad()
def eval_split(model, data_root, split, feature_fn, device, threshold=0.5):
    ds = AngryDataset(
        os.path.join(data_root, split),
        feature_fn=feature_fn,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    all_probs, all_labels = [], []
    for features, labels in loader:
        features = features.to(device)
        logits = model(features).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > threshold).astype(np.float32)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    # specificity / NPV
    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)

    # find best threshold by F1 on this split (for analysis only)
    best_t, best_f1 = threshold, f1
    for t in np.linspace(0.1, 0.9, 81):
        p = (probs > t).astype(np.float32)
        _tp = ((p == 1) & (labels == 1)).sum()
        _fp = ((p == 1) & (labels == 0)).sum()
        _fn = ((p == 0) & (labels == 1)).sum()
        prec = _tp / max(_tp + _fp, 1)
        rec = _tp / max(_tp + _fn, 1)
        _f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if _f1 > best_f1:
            best_f1, best_t = float(_f1), float(t)

    return {
        "split": split,
        "n": total,
        "n_angry": int(labels.sum()),
        "n_non_angry": int(total - labels.sum()),
        "threshold": threshold,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "npv": npv,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "best_threshold_f1": best_t,
        "best_f1_at_thresh": best_f1,
        "prob_mean_angry": float(probs[labels == 1].mean()) if (labels == 1).any() else 0.0,
        "prob_mean_non": float(probs[labels == 0].mean()) if (labels == 0).any() else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "best_model.pth"))
    parser.add_argument("--data_root", type=str, default=str(ROOT / "dataset"))
    parser.add_argument("--train_log", type=str, default="")
    parser.add_argument("--out", type=str, default=str(ROOT / "checkpoints" / "metrics_report.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", 1)
    dual = ckpt.get("dual", in_channels == 2)
    feature_fn = extract_dual_channel if dual else extract_mfcc

    model = TinyCNN(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    report = {
        "checkpoint": str(args.ckpt),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "in_channels": in_channels,
        "dual": dual,
        "params": count_parameters(model),
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_val_acc": ckpt.get("val_acc"),
        "ckpt_val_f1": ckpt.get("val_f1"),
        "splits": {},
        "history": [],
    }

    for split in ("train", "val", "test"):
        report["splits"][split] = eval_split(
            model, args.data_root, split, feature_fn, device, args.threshold
        )

    if args.train_log and Path(args.train_log).is_file():
        text = Path(args.train_log).read_text(encoding="utf-8", errors="ignore")
        report["history"] = parse_train_log(text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
