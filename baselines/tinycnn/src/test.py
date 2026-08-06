"""
Evaluate best checkpoint on test set.

Usage:
  python src/test.py
  python src/test.py --ckpt checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import AngryDataset
from feature import extract_dual_channel, extract_mfcc
from model import TinyCNN


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(ROOT / "dataset"))
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "best_model.pth"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", 1)
    dual = ckpt.get("dual", in_channels == 2)
    feature_fn = extract_dual_channel if dual else extract_mfcc

    test_ds = AngryDataset(
        os.path.join(args.data_root, "test"),
        feature_fn=feature_fn,
        augment=False,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = TinyCNN(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tp = fp = tn = fn = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        probs = torch.sigmoid(model(features).squeeze(-1))
        preds = (probs > args.threshold).float()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"Checkpoint: {args.ckpt}")
    print(f"Test N={total} | Acc={acc:.4f} | P={precision:.4f} | R={recall:.4f} | F1={f1:.4f}")
    print(f"Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    if "val_acc" in ckpt:
        print(f"Recorded val_acc={ckpt['val_acc']:.4f}, val_f1={ckpt.get('val_f1', float('nan')):.4f}")


if __name__ == "__main__":
    main()
