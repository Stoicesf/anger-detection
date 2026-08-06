"""Fine-tune DS-CNN on Chinese data + hard negatives (Phase 2).

Language: zh-CN only (CASIA / CNEV / hard_negative).
Keeps 5-class head; hard negatives mapped to non-anger classes.
Exports dscnn_casia_deploy.onnx for the realtime demo.

Usage:
  python scripts/build_hard_negative.py --clean
  python scripts/finetune_hard_negative.py --epochs 25 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from common import (
    HOP_LENGTH,
    MODELS_DIR,
    N_FFT,
    N_MELS,
    NUM_CLASSES,
    OUTPUT_DIR,
    SR,
    TIME_FRAMES,
    ensure_dirs,
)
from train_torch import DSCNN, set_seed

ROOT = Path(__file__).resolve().parents[1]
CASIA_ROOT = Path(r"E:\桌面\test\CASIA")
CNEV_ROOT = Path(r"E:\桌面\test\CNEV_Vocalizations\Core Set")
HN_ROOT = ROOT / "dataset" / "hard_negative"

FOLDER_TO_LABEL = {
    "angry": 2,
    "anger": 2,
    "happy": 1,
    "happiness": 1,
    "neutral": 0,
    "sad": 3,
    "sadness": 3,
    "surprise": 4,
}
HN_BUCKET_LABEL = {
    "loud_normal": 0,
    "speech": 0,
    "excited": 1,
    "surprise": 4,
    "fear": 0,
}
LABEL_NAMES = ["neutral", "happy", "angry", "sad", "surprise"]
CROPS = 4


def wav_to_mel(y: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt((y**2).mean()))
    if rms > 1e-4:
        y = y * (0.05 / rms)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    db = np.clip(librosa.power_to_db(mel, ref=1.0), -60.0, 20.0)
    return ((db + 60.0) / 80.0).T.astype(np.float32)


def load_mel(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return wav_to_mel(y.astype(np.float32))


def crop_patches(feat: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    t = feat.shape[0]
    out = []
    if t >= TIME_FRAMES:
        s0 = (t - TIME_FRAMES) // 2
        out.append(feat[s0 : s0 + TIME_FRAMES])
        for _ in range(CROPS - 1):
            s = int(rng.integers(0, t - TIME_FRAMES + 1))
            out.append(feat[s : s + TIME_FRAMES])
    else:
        pb = (TIME_FRAMES - t) // 2
        pad = np.pad(feat, ((pb, TIME_FRAMES - t - pb), (0, 0)), mode="edge")
        out = [pad] * CROPS
    return out


def collect_rows(use_cnev: bool, use_hn: bool) -> list[dict]:
    rows = []
    for folder in sorted(CASIA_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        lab = FOLDER_TO_LABEL.get(folder.name.lower())
        if lab is None:
            continue
        for wav in sorted(folder.glob("*.wav")):
            rows.append(
                {
                    "path": wav,
                    "label": lab,
                    "speaker": wav.stem.split("-")[-1],
                    "source": "CASIA",
                    "lang": "zh",
                }
            )
    if use_cnev:
        for gender in ("female", "male"):
            gdir = CNEV_ROOT / gender
            if not gdir.is_dir():
                continue
            for folder in gdir.iterdir():
                lab = FOLDER_TO_LABEL.get(folder.name.lower())
                if lab is None:
                    continue
                for wav in list(folder.glob("*.wav"))[:100]:
                    rows.append(
                        {
                            "path": wav,
                            "label": lab,
                            "speaker": f"cnev_{gender}",
                            "source": "CNEV",
                            "lang": "zh",
                        }
                    )
    if use_hn and (HN_ROOT / "manifest.json").exists():
        man = json.loads((HN_ROOT / "manifest.json").read_text(encoding="utf-8"))
        for it in man["items"]:
            bucket = it["bucket"]
            lab = HN_BUCKET_LABEL.get(bucket)
            if lab is None:
                continue
            rows.append(
                {
                    "path": ROOT / it["path"],
                    "label": lab,
                    "speaker": f"hn_{bucket}",
                    "source": "HN",
                    "lang": "zh",
                }
            )
    return rows


def augment_wave(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # gain
    if rng.random() < 0.5:
        y = y * float(rng.uniform(0.5, 2.0))
    # additive noise ~30%
    if rng.random() < 0.30:
        noise = rng.normal(0, 1, size=y.shape).astype(np.float32)
        p_sig = float(np.mean(y**2) + 1e-9)
        p_noi = float(np.mean(noise**2) + 1e-9)
        snr = float(rng.uniform(8, 22))
        scale = np.sqrt(p_sig / (p_noi * 10 ** (snr / 10)))
        y = y + noise * scale
    # speed ~20%
    if rng.random() < 0.20 and len(y) > SR // 2:
        rate = float(rng.choice([0.9, 0.95, 1.05, 1.1]))
        try:
            y = librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
        except Exception:
            pass
    return y.astype(np.float32)


class MelDS(Dataset):
    def __init__(self, X, y, train=False):
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
        return torch.from_numpy(np.ascontiguousarray(x).reshape(1, TIME_FRAMES, N_MELS)), int(
            self.y[i]
        )


def build_features(rows: list[dict], cache: Path, online_aug: bool, seed: int):
    if cache.exists() and not online_aug:
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], d["file_ids"], d["speakers"].tolist(), d["sources"].tolist()

    rng = np.random.default_rng(seed)
    Xs, ys, fids, spks, srcs = [], [], [], [], []
    for i, row in enumerate(rows):
        y, _ = librosa.load(row["path"], sr=SR, mono=True)
        y = y.astype(np.float32)
        if online_aug and row["source"] in ("CASIA", "CNEV", "HN"):
            # always keep clean crop + maybe one aug copy for HN/CASIA non-anger
            variants = [y]
            if row["label"] != 2 and rng.random() < 0.5:
                variants.append(augment_wave(y, rng))
        else:
            variants = [y]
        for v in variants:
            feat = wav_to_mel(v)
            for patch in crop_patches(feat, rng):
                Xs.append(patch.reshape(TIME_FRAMES, N_MELS, 1))
                ys.append(row["label"])
                fids.append(i)
                spks.append(row["speaker"])
                srcs.append(row["source"])
        if (i + 1) % 200 == 0 or i + 1 == len(rows):
            print(f"  features {i+1}/{len(rows)}", flush=True)

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    file_ids = np.asarray(fids, dtype=np.int64)
    speakers = np.asarray(spks, dtype=object)
    sources = np.asarray(srcs, dtype=object)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, X=X, y=y, file_ids=file_ids, speakers=speakers, sources=sources
    )
    return X, y, file_ids, spks, srcs


def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            yt.append(yb.numpy())
            yp.append(pred)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    acc = float((yt == yp).mean())
    # binary angry metrics
    yt_b = (yt == 2).astype(int)
    yp_b = (yp == 2).astype(int)
    tp = int(((yp_b == 1) & (yt_b == 1)).sum())
    fp = int(((yp_b == 1) & (yt_b == 0)).sum())
    fn = int(((yp_b == 0) & (yt_b == 1)).sum())
    tn = int(((yp_b == 0) & (yt_b == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "acc5": acc,
        "angry_precision": prec,
        "angry_recall": rec,
        "angry_f1": f1,
        "angry_specificity": tn / max(tn + fp, 1),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-speaker", default="ZhaoZuoxiang")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-cnev", action="store_true")
    ap.add_argument("--no-hn", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    set_seed(42)
    device = torch.device(args.device)
    print("device", device, "| language=zh-CN only")

    rows = collect_rows(use_cnev=not args.no_cnev, use_hn=not args.no_hn)
    print(
        "files",
        len(rows),
        "by source",
        {s: sum(1 for r in rows if r["source"] == s) for s in ("CASIA", "CNEV", "HN")},
    )
    # enforce Chinese-only
    rows = [r for r in rows if r.get("lang") == "zh"]

    cache = OUTPUT_DIR / "hn_finetune_cache.npz"
    if args.rebuild_cache and cache.exists():
        cache.unlink()
    X, y, file_ids, speakers, sources = build_features(
        rows, cache, online_aug=True, seed=42
    )
    speakers = list(speakers)
    sources = list(sources)

    # held-out CASIA speaker for test; HN/CNEV always train
    test_mask = np.array(
        [
            (sp == args.test_speaker and src == "CASIA")
            for sp, src in zip(speakers, sources)
        ]
    )
    train_mask = ~test_mask
    train_fids = np.unique(file_ids[train_mask])
    rng = np.random.default_rng(42)
    rng.shuffle(train_fids)
    n_val = max(1, int(0.1 * len(train_fids)))
    val_fids = set(train_fids[:n_val].tolist())
    final_train = np.array(
        [m and int(fid) not in val_fids for m, fid in zip(train_mask, file_ids)]
    )
    final_val = np.array(
        [m and int(fid) in val_fids for m, fid in zip(train_mask, file_ids)]
    )

    Xtr, ytr = X[final_train], y[final_train]
    Xva, yva = X[final_val], y[final_val]
    Xte, yte = X[test_mask], y[test_mask]
    print(f"crops train={len(Xtr)} val={len(Xva)} test={len(Xte)}")
    print("train dist", np.bincount(ytr, minlength=5).tolist())

    # class weights; boost non-anger a bit for lower FAR (up-weight classes 0,1,3,4 slightly vs angry)
    counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))
    weights = np.clip(weights, None, 5.0)
    # slightly down-weight angry to reduce over-trigger in wild
    weights[2] *= 0.85
    weights = weights / weights.mean()

    train_loader = DataLoader(
        MelDS(Xtr, ytr, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(MelDS(Xva, yva), batch_size=128, shuffle=False)
    test_loader = DataLoader(MelDS(Xte, yte), batch_size=128, shuffle=False)

    model = DSCNN().to(device)
    pretrained = MODELS_DIR / "dscnn_casia_deploy.pt"
    if not pretrained.exists():
        pretrained = MODELS_DIR / "dscnn_torch_best.pt"
    model.load_state_dict(torch.load(pretrained, map_location=device, weights_only=True))
    print("loaded", pretrained)

    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_state, wait = -1.0, None, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        sch.step()
        met = evaluate(model, val_loader, device)
        # optimize angry F1 with specificity preference
        score = 0.5 * met["angry_f1"] + 0.3 * met["angry_specificity"] + 0.2 * met["acc5"]
        print(
            f"epoch {epoch:02d} train={correct/max(total,1):.4f} "
            f"val_acc5={met['acc5']:.4f} angryF1={met['angry_f1']:.4f} "
            f"spec={met['angry_specificity']:.4f} score={score:.4f} "
            f"({(time.time()-t0)/60:.1f}m)",
            flush=True,
        )
        if score >= best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 8:
                print("early stop")
                break

    model.load_state_dict(best_state)
    test_met = evaluate(model, test_loader, device)
    print("\n=== Held-out Chinese speaker test ===")
    print(json.dumps(test_met, indent=2))

    # deploy: train a bit more on all except we keep deploy = best + short all-train
    print("\n=== Deploy pass on all zh data ===")
    all_mask = np.ones(len(X), dtype=bool)
    all_loader = DataLoader(
        MelDS(X[all_mask], y[all_mask], train=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    deploy = DSCNN().to(device)
    deploy.load_state_dict(best_state)
    opt2 = torch.optim.AdamW(deploy.parameters(), lr=args.lr * 0.4, weight_decay=1e-4)
    for epoch in range(8):
        deploy.train()
        tot, cor = 0, 0
        for xb, yb in all_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt2.zero_grad()
            out = deploy(xb)
            loss = crit(out, yb)
            loss.backward()
            opt2.step()
            cor += (out.argmax(1) == yb).sum().item()
            tot += len(yb)
        print(f"deploy {epoch:02d} train={cor/max(tot,1):.4f}", flush=True)

    torch.save(deploy.state_dict(), MODELS_DIR / "dscnn_casia_deploy.pt")
    torch.save(best_state, MODELS_DIR / "dscnn_casia_best.pt")
    export = nn.Sequential(deploy.cpu(), nn.Softmax(dim=1)).eval()
    torch.onnx.export(
        export,
        torch.randn(1, 1, TIME_FRAMES, N_MELS),
        str(MODELS_DIR / "dscnn_casia_deploy.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    export_b = nn.Sequential(DSCNN(), nn.Softmax(dim=1)).eval()
    export_b[0].load_state_dict(best_state)
    torch.onnx.export(
        export_b,
        torch.randn(1, 1, TIME_FRAMES, N_MELS),
        str(MODELS_DIR / "dscnn_casia.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )

    metrics = {
        "language": "zh-CN",
        "phase": 2,
        "test_speaker": args.test_speaker,
        "test": test_met,
        "best_score": best,
        "n_rows": len(rows),
        "use_hard_negative": not args.no_hn,
    }
    (OUTPUT_DIR / "hn_finetune_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved models +", OUTPUT_DIR / "hn_finetune_metrics.json")


if __name__ == "__main__":
    main()
