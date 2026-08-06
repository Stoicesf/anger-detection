"""
Train TinyCNN for angry detection.

Usage (from project root, conda env pytorch12):
  python src/train.py
  python src/train.py --dual --epochs 80 --batch_size 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import AngryDataset
from feature import extract_dual_channel, extract_mfcc
from model import TinyCNN, count_parameters


def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features).squeeze(-1)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "loss": avg_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(ROOT / "dataset"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows-safe default
    parser.add_argument("--dual", action="store_true", help="use MFCC+F0 dual channel")
    parser.add_argument("--augment", action="store_true", help="train-time audio augment")
    parser.add_argument("--ckpt_dir", type=str, default=str(ROOT / "checkpoints"))
    parser.add_argument("--patience", type=int, default=15, help="early stop patience")
    args = parser.parse_args()

    feature_fn = extract_dual_channel if args.dual else extract_mfcc
    in_channels = 2 if args.dual else 1

    train_ds = AngryDataset(
        os.path.join(args.data_root, "train"),
        feature_fn=feature_fn,
        augment=args.augment,
    )
    val_ds = AngryDataset(
        os.path.join(args.data_root, "val"),
        feature_fn=feature_fn,
        augment=False,
    )

    n_pos, n_neg = train_ds.class_counts()
    print(f"Train samples: angry={n_pos}, non_angry={n_neg}, total={len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = TinyCNN(in_channels=in_channels).to(device)
    print(f"Params: {count_parameters(model)}")

    # class imbalance: angry << non_angry
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pth"
    history_path = ckpt_dir / "train_history.json"

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    bad_epochs = 0
    history = []
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            n_seen += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running / max(n_seen, 1)
        metrics = evaluate(model, val_loader, device, criterion)
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_acc": metrics["acc"],
            "val_f1": metrics["f1"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "lr": lr,
        }
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"val_acc={metrics['acc']:.4f} | "
            f"val_f1={metrics['f1']:.4f} | "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}",
            flush=True,
        )

        # prefer F1 under imbalance; also track acc
        score = metrics["f1"]
        if score > best_val_f1 or (
            abs(score - best_val_f1) < 1e-6 and metrics["acc"] > best_val_acc
        ):
            best_val_f1 = score
            best_val_acc = metrics["acc"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_channels": in_channels,
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "val_f1": best_val_f1,
                    "dual": args.dual,
                    "augment": args.augment,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  -> saved {best_path} (f1={best_val_f1:.4f})", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stop at epoch {epoch} (patience={args.patience})", flush=True)
                break

    elapsed = time.perf_counter() - t0
    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "epochs_ran": len(history),
        "elapsed_sec": elapsed,
        "params": count_parameters(model),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dual": args.dual,
        "augment": args.augment,
    }
    (ckpt_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"Best Val Acc={best_val_acc:.4f}, Best Val F1={best_val_f1:.4f}, "
        f"epoch={best_epoch}, elapsed={elapsed/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
