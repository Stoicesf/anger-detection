"""Fine-tune DS-CNN on CASIA (+ optional CNEV) for better real-world emotion accuracy.

Loads pretrained dscnn_torch_best.pt, trains with speaker-held-out validation,
exports ONNX for the realtime demo.

Usage:
  conda activate pytorch12
  python scripts/finetune_casia.py
  python scripts/finetune_casia.py --test-speaker ZhaoZuoxiang --epochs 40
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from anger_detection.common import (
    CLASS_NAMES,
    HOP_LENGTH,
    MODELS_DIR,
    N_FFT,
    N_MELS,
    NUM_CLASSES,
    OUTPUT_DIR,
    SR,
    TIME_FRAMES,
    CASIA_ROOT,
    CNEV_ROOT,
    ensure_dirs,
)
from anger_detection.models import DSCNN, set_seed

FOLDER_TO_LABEL = {
    "angry": 2,
    "anger": 2,
    "happy": 1,
    "happiness": 1,
    "neutral": 0,
    "sad": 3,
    "sadness": 3,
    "surprise": 4,
    # fear intentionally skipped (model has 5 classes)
}

CROPS_PER_FILE = 4
LABEL_NAMES = ["neutral", "happy", "angry", "sad", "surprise"]


def log_mel_from_path(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    rms = float(np.sqrt((y**2).mean()))
    if rms > 1e-4:
        y = y * (0.05 / rms)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    db = librosa.power_to_db(mel, ref=1.0)
    db = np.clip(db, -60.0, 20.0)
    return ((db + 60.0) / 80.0).T.astype(np.float32)  # (T, 40)


def crop_patches(feat: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    t = feat.shape[0]
    patches = []
    if t >= TIME_FRAMES:
        start = (t - TIME_FRAMES) // 2
        patches.append(feat[start : start + TIME_FRAMES])
        for _ in range(CROPS_PER_FILE - 1):
            s = int(rng.integers(0, t - TIME_FRAMES + 1))
            patches.append(feat[s : s + TIME_FRAMES])
    else:
        pad_before = (TIME_FRAMES - t) // 2
        pad_after = TIME_FRAMES - t - pad_before
        padded = np.pad(feat, ((pad_before, pad_after), (0, 0)), mode="edge")
        patches = [padded] * CROPS_PER_FILE
    return patches


def collect_casia() -> list[dict]:
    rows = []
    for folder in sorted(CASIA_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        label = FOLDER_TO_LABEL.get(folder.name.lower())
        if label is None:
            continue
        for wav in sorted(folder.glob("*.wav")):
            speaker = wav.stem.split("-")[-1]
            rows.append(
                {
                    "path": wav,
                    "label": label,
                    "speaker": speaker,
                    "source": "CASIA",
                }
            )
    return rows


def collect_cnev(limit_per_class: int = 80) -> list[dict]:
    by_label: dict[int, list[Path]] = defaultdict(list)
    for gender in ("female", "male"):
        gdir = CNEV_ROOT / gender
        if not gdir.is_dir():
            continue
        for folder in gdir.iterdir():
            if not folder.is_dir():
                continue
            label = FOLDER_TO_LABEL.get(folder.name.lower())
            if label is None:
                continue
            for wav in folder.glob("*.wav"):
                by_label[label].append(wav)
    rng = np.random.default_rng(42)
    rows = []
    for label, paths in by_label.items():
        paths = list(paths)
        rng.shuffle(paths)
        for wav in paths[:limit_per_class]:
            rows.append(
                {
                    "path": wav,
                    "label": label,
                    "speaker": f"cnev_{wav.parent.parent.name}",
                    "source": "CNEV",
                }
            )
    return rows


def build_features(rows: list[dict], cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        print(f"Loaded cache: {cache_path}")
        return data["X"], data["y"], data["file_ids"], data["speakers"].tolist()

    rng = np.random.default_rng(42)
    Xs, ys, fids, speakers = [], [], [], []
    for i, row in enumerate(rows):
        feat = log_mel_from_path(row["path"])
        for patch in crop_patches(feat, rng):
            Xs.append(patch.reshape(TIME_FRAMES, N_MELS, 1))
            ys.append(row["label"])
            fids.append(i)
            speakers.append(row["speaker"])
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"  features {i+1}/{len(rows)}", flush=True)

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    file_ids = np.asarray(fids, dtype=np.int64)
    speakers_arr = np.asarray(speakers, dtype=object)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, file_ids=file_ids, speakers=speakers_arr)
    print(f"Saved cache -> {cache_path}  X={X.shape}")
    return X, y, file_ids, speakers


class MelDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, train: bool = False):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i, :, :, 0]
        if self.train:
            pad = 12
            x = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
            off = np.random.randint(0, 2 * pad + 1)
            x = x[off : off + TIME_FRAMES]
            x = x + np.random.normal(0, 0.003, x.shape).astype(np.float32)
            if np.random.rand() < 0.3:
                x = np.clip(x * np.random.uniform(0.85, 1.15), 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x).reshape(1, TIME_FRAMES, N_MELS)), int(self.y[i])


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).argmax(1).cpu().numpy()
            ys.append(yb.numpy())
            ps.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    return acc, macro_f1, y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-speaker", default="ZhaoZuoxiang", help="held-out CASIA speaker")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="cpu recommended while GPU is busy")
    parser.add_argument("--use-cnev", action="store_true", default=True)
    parser.add_argument("--no-cnev", action="store_true")
    parser.add_argument("--train-all", action="store_true", help="also train deploy model on all CASIA")
    args = parser.parse_args()
    if args.no_cnev:
        args.use_cnev = False

    ensure_dirs()
    set_seed(42)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print("device:", device)

    rows = collect_casia()
    print(f"CASIA usable files: {len(rows)}")
    if args.use_cnev:
        cnev = collect_cnev(80)
        print(f"CNEV added: {len(cnev)}")
        rows = rows + cnev

    cache = OUTPUT_DIR / "casia_finetune_cache.npz"
    X, y, file_ids, speakers = build_features(rows, cache)

    # Speaker-held-out split (CASIA only for test; CNEV always train)
    test_mask = np.array(
        [sp == args.test_speaker and rows[int(fid)]["source"] == "CASIA" for fid, sp in zip(file_ids, speakers)]
    )
    # Also keep file-level: unique CASIA files from other speakers for val
    train_mask = ~test_mask
    # Split remaining into train/val by file id (10% val)
    train_fids = np.unique(file_ids[train_mask])
    rng = np.random.default_rng(42)
    rng.shuffle(train_fids)
    n_val = max(1, int(0.1 * len(train_fids)))
    val_fids = set(train_fids[:n_val].tolist())
    final_train = np.array([m and int(fid) not in val_fids for m, fid in zip(train_mask, file_ids)])
    final_val = np.array([m and int(fid) in val_fids for m, fid in zip(train_mask, file_ids)])

    X_train, y_train = X[final_train], y[final_train]
    X_val, y_val = X[final_val], y[final_val]
    X_test, y_test = X[test_mask], y[test_mask]
    print(
        f"crops train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"(test speaker={args.test_speaker})"
    )
    print("train dist", np.bincount(y_train, minlength=NUM_CLASSES).tolist())
    print("test  dist", np.bincount(y_test, minlength=NUM_CLASSES).tolist())

    train_loader = DataLoader(
        MelDataset(X_train, y_train, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        MelDataset(X_val, y_val, train=False),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        MelDataset(X_test, y_test, train=False),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    model = DSCNN().to(device)
    pretrained = MODELS_DIR / "dscnn_torch_best.pt"
    if pretrained.exists():
        state = torch.load(pretrained, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print("Loaded pretrained:", pretrained)
    else:
        print("WARNING: no pretrained weights, training from scratch")

    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))
    weights = np.clip(weights, None, 5.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val, best_state, wait = 0.0, None, 0
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        scheduler.step()
        train_acc = correct / max(total, 1)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch {epoch:02d} train={train_acc:.4f} val={val_acc:.4f} "
            f"f1={val_f1:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"({(time.time()-t0)/60:.1f}m)",
            flush=True,
        )
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 10:
                print(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_acc, test_f1, yt, yp = evaluate(model, test_loader, device)
    print("\n=== Held-out speaker test ===")
    print(f"Acc={test_acc:.4f} Macro-F1={test_f1:.4f}")
    print(classification_report(yt, yp, target_names=LABEL_NAMES, digits=4))
    print("CM\n", confusion_matrix(yt, yp))

    out_pt = MODELS_DIR / "dscnn_casia_best.pt"
    torch.save(model.state_dict(), out_pt)

    # Export ONNX with softmax
    export = nn.Sequential(model.cpu(), nn.Softmax(dim=1)).eval()
    onnx_path = MODELS_DIR / "dscnn_casia.onnx"
    torch.onnx.export(
        export,
        torch.randn(1, 1, TIME_FRAMES, N_MELS),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    print("Saved", out_pt)
    print("Saved", onnx_path)

    # Optional: train-all deploy model (better for demo on known speakers)
    if args.train_all or True:
        print("\n=== Train deploy model on all CASIA(+CNEV) ===")
        all_loader = DataLoader(
            MelDataset(X[train_mask | test_mask], y[train_mask | test_mask], train=True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        deploy = DSCNN().to(device)
        if pretrained.exists():
            deploy.load_state_dict(torch.load(pretrained, map_location=device, weights_only=True))
        # warm-start from speaker-held-out best too
        deploy.load_state_dict(best_state)
        deploy.to(device)
        opt2 = torch.optim.AdamW(deploy.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)
        crit2 = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
        for epoch in range(12):
            deploy.train()
            total, correct = 0, 0
            for xb, yb in all_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt2.zero_grad()
                out = deploy(xb)
                loss = crit2(out, yb)
                loss.backward()
                opt2.step()
                correct += (out.argmax(1) == yb).sum().item()
                total += len(yb)
            print(f"deploy epoch {epoch:02d} train_acc={correct/max(total,1):.4f}", flush=True)

        deploy_pt = MODELS_DIR / "dscnn_casia_deploy.pt"
        torch.save(deploy.state_dict(), deploy_pt)
        export2 = nn.Sequential(deploy.cpu(), nn.Softmax(dim=1)).eval()
        deploy_onnx = MODELS_DIR / "dscnn_casia_deploy.onnx"
        torch.onnx.export(
            export2,
            torch.randn(1, 1, TIME_FRAMES, N_MELS),
            str(deploy_onnx),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )
        print("Saved", deploy_pt, deploy_onnx)

    metrics = {
        "test_speaker": args.test_speaker,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "best_val_accuracy": float(best_val),
        "epochs": len(history),
        "device": str(device),
        "class_names": CLASS_NAMES,
        "n_train_crops": int(len(X_train)),
        "n_test_crops": int(len(X_test)),
        "use_cnev": args.use_cnev,
        "history": history,
        "baseline_casia_acc_before": 0.341,
    }
    metrics_path = OUTPUT_DIR / "casia_finetune_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Metrics ->", metrics_path)


if __name__ == "__main__":
    main()
