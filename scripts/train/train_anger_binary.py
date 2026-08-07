"""Anger Detection V2: binary DSCNN trainer (dscnn_anger_v2).

Changes vs finetune_hard_negative.py (kept as baseline):
  - binary labels: anger=1 / others=0
  - all hard-negatives mapped to 0 (no happy/surprise soft labels)
  - crops: 1 center + 2 random + 2 tail (family outburst at utterance end)
  - WeightedRandomSampler (anger weight 3)
  - Binary Focal Loss
  - milder gain + optional pitch shift
  - optional SE (--se)

Usage:
  conda activate pytorch12
  python scripts/train/train_anger_binary.py --device cuda --epochs 40
  python scripts/train/train_anger_binary.py --device cuda --epochs 40 --se
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from anger_detection.common import (
    CURATED_DIR,
    HOP_LENGTH,
    MODELS_DIR,
    N_FFT,
    N_MELS,
    OUTPUT_DIR,
    SR,
    TIME_FRAMES,
    CASIA_ROOT,
    CNEV_ROOT,
    PROJECT_ROOT,
    ensure_dirs,
    resolve_curated_path,
)
from anger_detection.losses import BinaryFocalLoss
from anger_detection.models.model_anger import DSCNN_Anger, DSCNN_AngerSigmoid, load_backbone_from_5class
from anger_detection.models import set_seed

HN_ROOT = CURATED_DIR / "hard_negative"
HOME_ROOT = CURATED_DIR / "home_anger"

# Original 5-class folder map (only used before binary conversion)
FOLDER_TO_5 = {
    "angry": 2,
    "anger": 2,
    "happy": 1,
    "happiness": 1,
    "neutral": 0,
    "sad": 3,
    "sadness": 3,
    "surprise": 4,
}
# V2: every HN bucket is a hard *negative* (not happy/surprise)
HN_BUCKET_BINARY = {
    "loud_normal": 0,
    "speech": 0,
    "excited": 0,
    "surprise": 0,
    "fear": 0,
}
CROPS = 5  # 1 center + 2 random + 2 tail
ANGER_SAMPLER_WEIGHT = 3.0


def to_binary(label5: int) -> int:
    return 1 if int(label5) == 2 else 0


def wav_to_mel(y: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt((y**2).mean()))
    if rms > 1e-4:
        y = y * (0.05 / rms)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    db = np.clip(librosa.power_to_db(mel, ref=1.0), -60.0, 20.0)
    return ((db + 60.0) / 80.0).T.astype(np.float32)


def crop_patches(feat: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """1 center + 2 random + 2 tail (anger often near utterance end)."""
    t = feat.shape[0]
    out: list[np.ndarray] = []
    if t < TIME_FRAMES:
        pb = (TIME_FRAMES - t) // 2
        pad = np.pad(feat, ((pb, TIME_FRAMES - t - pb), (0, 0)), mode="edge")
        return [pad] * CROPS

    # center
    s0 = (t - TIME_FRAMES) // 2
    out.append(feat[s0 : s0 + TIME_FRAMES])

    # 2 random
    for _ in range(2):
        s = int(rng.integers(0, t - TIME_FRAMES + 1))
        out.append(feat[s : s + TIME_FRAMES])

    # 2 tail (end-aligned + slight jitter)
    s_tail = t - TIME_FRAMES
    out.append(feat[s_tail : s_tail + TIME_FRAMES])
    if s_tail > 0:
        jitter = int(rng.integers(0, min(8, s_tail) + 1))
        s2 = max(0, s_tail - jitter)
        out.append(feat[s2 : s2 + TIME_FRAMES])
    else:
        out.append(feat[s_tail : s_tail + TIME_FRAMES])
    return out


def augment_wave(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # milder gain (V2): avoid happy→anger loudness artifacts
    if rng.random() < 0.5:
        y = y * float(rng.uniform(0.8, 1.5))
    if rng.random() < 0.30:
        noise = rng.normal(0, 1, size=y.shape).astype(np.float32)
        p_sig = float(np.mean(y**2) + 1e-9)
        p_noi = float(np.mean(noise**2) + 1e-9)
        snr = float(rng.uniform(10, 24))
        scale = np.sqrt(p_sig / (p_noi * 10 ** (snr / 10)))
        y = y + noise * scale
    if rng.random() < 0.20 and len(y) > SR // 2:
        rate = float(rng.choice([0.95, 1.05]))
        try:
            y = librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
        except Exception:
            pass
    # pitch ±2 semitone (~parent age / gender diversity)
    if rng.random() < 0.25 and len(y) > SR // 2:
        n_steps = float(rng.choice([-2, -1, 1, 2]))
        try:
            y = librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps).astype(np.float32)
        except Exception:
            pass
    return y.astype(np.float32)


def _iter_home_wavs(root: Path) -> list[dict]:
    rows = []
    if not root.is_dir():
        return rows
    pos = root / "positive"
    neg = root / "negative"
    if pos.is_dir():
        for wav in pos.rglob("*.wav"):
            rows.append(
                {
                    "path": wav,
                    "label": 1,
                    "speaker": f"home_{wav.parent.name}",
                    "source": "HOME",
                    "lang": "zh",
                }
            )
    if neg.is_dir():
        for wav in neg.rglob("*.wav"):
            rows.append(
                {
                    "path": wav,
                    "label": 0,
                    "speaker": f"home_{wav.parent.name}",
                    "source": "HOME",
                    "lang": "zh",
                }
            )
    return rows


def collect_rows(use_cnev: bool, use_hn: bool, use_home: bool) -> list[dict]:
    rows: list[dict] = []
    for folder in sorted(CASIA_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        lab5 = FOLDER_TO_5.get(folder.name.lower())
        if lab5 is None:
            continue
        for wav in sorted(folder.glob("*.wav")):
            rows.append(
                {
                    "path": wav,
                    "label": to_binary(lab5),
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
                lab5 = FOLDER_TO_5.get(folder.name.lower())
                if lab5 is None:
                    continue
                for wav in list(folder.glob("*.wav"))[:100]:
                    rows.append(
                        {
                            "path": wav,
                            "label": to_binary(lab5),
                            "speaker": f"cnev_{gender}",
                            "source": "CNEV",
                            "lang": "zh",
                        }
                    )
    if use_hn and (HN_ROOT / "manifest.json").exists():
        man = json.loads((HN_ROOT / "manifest.json").read_text(encoding="utf-8"))
        for it in man["items"]:
            bucket = it["bucket"]
            lab = HN_BUCKET_BINARY.get(bucket)
            if lab is None:
                continue
            rows.append(
                {
                    "path": resolve_curated_path(it["path"]),
                    "label": lab,
                    "speaker": f"hn_{bucket}",
                    "source": "HN",
                    "lang": "zh",
                }
            )
    if use_home:
        rows.extend(_iter_home_wavs(HOME_ROOT))
    return rows


class MelDS(Dataset):
    def __init__(self, X, y, train: bool = False):
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
        x = torch.from_numpy(np.ascontiguousarray(x).reshape(1, TIME_FRAMES, N_MELS))
        y = torch.tensor(float(self.y[i]), dtype=torch.float32)
        return x, y


def build_features(rows: list[dict], cache: Path, online_aug: bool, seed: int):
    if cache.exists() and not online_aug:
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], d["file_ids"], d["speakers"].tolist(), d["sources"].tolist()

    rng = np.random.default_rng(seed)
    Xs, ys, fids, spks, srcs = [], [], [], [], []
    for i, row in enumerate(rows):
        y, _ = librosa.load(row["path"], sr=SR, mono=True)
        y = y.astype(np.float32)
        variants = [y]
        # mild aug copies: prefer anger oversampling via sampler; light aug on both
        if online_aug and rng.random() < (0.6 if row["label"] == 1 else 0.35):
            variants.append(augment_wave(y, rng))
        for v in variants:
            feat = wav_to_mel(v)
            for patch in crop_patches(feat, rng):
                Xs.append(patch.reshape(TIME_FRAMES, N_MELS, 1))
                ys.append(row["label"])
                fids.append(i)
                spks.append(row["speaker"])
                srcs.append(row["source"])
        if (i + 1) % 200 == 0 or i + 1 == len(rows):
            print(f"  features {i + 1}/{len(rows)}", flush=True)

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


def binary_metrics(yt: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> dict:
    yp = (scores >= thr).astype(np.int64)
    yt = yt.astype(np.int64)
    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, average="binary", zero_division=0
    )
    out = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "threshold": thr,
    }
    if len(np.unique(yt)) > 1:
        out["auc"] = float(roc_auc_score(yt, scores))
        out["pr_auc"] = float(average_precision_score(yt, scores))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


@torch.no_grad()
def evaluate(model, loader, device, thr: float = 0.5) -> dict:
    model.eval()
    yt, scores, loss_sum, n = [], [], 0.0, 0
    crit = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logit = model(xb).view(-1)
        loss_sum += crit(logit, yb).item() * len(yb)
        n += len(yb)
        yt.append(yb.cpu().numpy())
        scores.append(torch.sigmoid(logit).cpu().numpy())
    yt = np.concatenate(yt)
    scores = np.concatenate(scores)
    met = binary_metrics(yt, scores, thr=thr)
    met["loss"] = loss_sum / max(n, 1)
    met["n_pos"] = int(yt.sum())
    met["n_neg"] = int(len(yt) - yt.sum())
    return met


def make_sampler(y: np.ndarray) -> WeightedRandomSampler:
    y = np.asarray(y)
    w = np.where(y == 1, ANGER_SAMPLER_WEIGHT, 1.0).astype(np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(w),
        replacement=True,
    )


def pick_pretrained() -> Path | None:
    for p in (
        MODELS_DIR / "dscnn_casia_deploy.pt",
        MODELS_DIR / "dscnn_casia_best.pt",
        MODELS_DIR / "dscnn_torch_best.pt",
    ):
        if p.exists():
            return p
    return None


def export_onnx(model: DSCNN_Anger, path: Path) -> None:
    wrapped = DSCNN_AngerSigmoid(model.cpu()).eval()
    dummy = torch.randn(1, 1, TIME_FRAMES, N_MELS)
    torch.onnx.export(
        wrapped,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["anger_prob"],
        opset_version=17,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-speaker", default="ZhaoZuoxiang")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-cnev", action="store_true")
    ap.add_argument("--no-hn", action="store_true")
    ap.add_argument("--use-home", action="store_true", help="include dataset/home_anger")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--se", action="store_true", help="enable SE attention (Day3 A/B)")
    ap.add_argument("--focal-alpha", type=float, default=0.75)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--thr", type=float, default=0.5, help="val metric threshold")
    ap.add_argument("--tag", default="v2", help="filename tag, e.g. v2 / v2_se")
    args = ap.parse_args()

    ensure_dirs()
    set_seed(42)
    device = torch.device(args.device)
    tag = args.tag if not args.se else (args.tag if "se" in args.tag else f"{args.tag}_se")
    print(f"device={device} | binary anger V2 | se={args.se} | tag={tag}")

    rows = collect_rows(
        use_cnev=not args.no_cnev, use_hn=not args.no_hn, use_home=args.use_home
    )
    rows = [r for r in rows if r.get("lang") == "zh"]
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = len(rows) - n_pos
    print(
        f"files={len(rows)} pos={n_pos} neg={n_neg} | by source",
        {s: sum(1 for r in rows if r["source"] == s) for s in ("CASIA", "CNEV", "HN", "HOME")},
    )

    cache = OUTPUT_DIR / f"anger_binary_{tag}_cache.npz"
    if args.rebuild_cache and cache.exists():
        cache.unlink()
    X, y, file_ids, speakers, sources = build_features(
        rows, cache, online_aug=True, seed=42
    )
    speakers, sources = list(speakers), list(sources)

    test_mask = np.array(
        [(sp == args.test_speaker and src == "CASIA") for sp, src in zip(speakers, sources)]
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
    print(
        f"crops train={len(Xtr)} val={len(Xva)} test={len(Xte)} | "
        f"train pos={int(ytr.sum())} neg={int(len(ytr) - ytr.sum())}"
    )

    train_ds = MelDS(Xtr, ytr, train=True)
    sampler = make_sampler(ytr)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(MelDS(Xva, yva), batch_size=128, shuffle=False)
    test_loader = DataLoader(MelDS(Xte, yte), batch_size=128, shuffle=False)

    model = DSCNN_Anger(use_se=args.se).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")

    pretrained = pick_pretrained()
    if pretrained is not None and not args.se:
        sd = torch.load(pretrained, map_location=device, weights_only=True)
        loaded, missing, _ = load_backbone_from_5class(model, sd)
        print(f"loaded backbone from {pretrained.name}: {len(loaded)} tensors, missing={len(missing)}")
    elif pretrained is not None and args.se:
        # SE inserts layers: load only conv1/bn0 and matching DSBlock weights by remapping
        sd = torch.load(pretrained, map_location="cpu", weights_only=True)
        remapped = {}
        # blocks in 5-class: 0,1,2,3 ; with SE: 0(ds),1(se),2(ds),3(se),...
        for k, v in sd.items():
            if k.startswith("fc."):
                continue
            if k.startswith("conv1.") or k.startswith("bn0."):
                remapped[k] = v
                continue
            if k.startswith("blocks."):
                parts = k.split(".")
                bi = int(parts[1])
                new_bi = bi * 2  # DSBlock indices when SE interleaved
                nk = ".".join(["blocks", str(new_bi)] + parts[2:])
                if nk in model.state_dict() and model.state_dict()[nk].shape == v.shape:
                    remapped[nk] = v
        miss = model.load_state_dict(remapped, strict=False)
        print(f"SE transfer from {pretrained.name}: {len(remapped)} tensors, missing={len(miss.missing_keys)}")
    else:
        print("no pretrained found, train from scratch")

    crit = BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_state, wait = -1.0, None, 0
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = model(xb)
            loss = crit(logit, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            n += len(yb)
        sch.step()
        train_loss = loss_sum / max(n, 1)
        val = evaluate(model, val_loader, device, thr=args.thr)
        # optimize F1 with mild AUC preference
        score = 0.7 * val["f1"] + 0.3 * (0.0 if np.isnan(val["auc"]) else val["auc"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
            "val_auc": val["auc"],
            "val_pr_auc": val["pr_auc"],
            "score": score,
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"P={val['precision']:.3f} R={val['recall']:.3f} F1={val['f1']:.3f} "
            f"AUC={val['auc']:.3f} PR-AUC={val['pr_auc']:.3f} score={score:.4f} "
            f"({(time.time() - t0) / 60:.1f}m)",
            flush=True,
        )
        if score >= best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 10:
                print("early stop")
                break

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, thr=args.thr)
    print("\n=== Held-out speaker test (binary) ===")
    print(json.dumps(test, indent=2))

    # short deploy pass on all data
    print("\n=== Deploy pass ===")
    all_loader = DataLoader(
        MelDS(X, y, train=True),
        batch_size=args.batch_size,
        sampler=make_sampler(y),
        drop_last=True,
    )
    deploy = DSCNN_Anger(use_se=args.se).to(device)
    deploy.load_state_dict(best_state)
    opt2 = torch.optim.AdamW(deploy.parameters(), lr=args.lr * 0.4, weight_decay=1e-4)
    for epoch in range(8):
        deploy.train()
        loss_sum, n = 0.0, 0
        for xb, yb in all_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt2.zero_grad()
            loss = crit(deploy(xb), yb)
            loss.backward()
            opt2.step()
            loss_sum += loss.item() * len(yb)
            n += len(yb)
        print(f"deploy {epoch:02d} loss={loss_sum / max(n, 1):.4f}", flush=True)

    pt_best = MODELS_DIR / f"dscnn_anger_{tag}_best.pt"
    pt_deploy = MODELS_DIR / f"dscnn_anger_{tag}.pt"
    onnx_path = MODELS_DIR / f"dscnn_anger_{tag}.onnx"
    # canonical names for default tag
    if tag in ("v2", "anger_v2"):
        pt_deploy = MODELS_DIR / "dscnn_anger_v2.pt"
        onnx_path = MODELS_DIR / "dscnn_anger_v2.onnx"
        pt_best = MODELS_DIR / "dscnn_anger_v2_best.pt"

    torch.save(best_state, pt_best)
    torch.save(deploy.state_dict(), pt_deploy)
    export_onnx(deploy, onnx_path)
    # also export best
    best_model = DSCNN_Anger(use_se=args.se)
    best_model.load_state_dict(best_state)
    export_onnx(best_model, MODELS_DIR / f"dscnn_anger_{tag}_best.onnx")

    metrics = {
        "version": "dscnn_anger_v2",
        "tag": tag,
        "use_se": args.se,
        "test_speaker": args.test_speaker,
        "test": test,
        "best_score": best,
        "n_files": len(rows),
        "n_pos_files": n_pos,
        "n_neg_files": n_neg,
        "crops_train_pos": int(ytr.sum()),
        "crops_train_neg": int(len(ytr) - ytr.sum()),
        "params": n_params,
        "history": history,
        "artifacts": {
            "pt_best": str(pt_best),
            "pt_deploy": str(pt_deploy),
            "onnx": str(onnx_path),
        },
    }
    out_json = OUTPUT_DIR / f"anger_binary_{tag}_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / f"anger_binary_{tag}_log.txt").write_text(
        "\n".join(
            f"epoch {h['epoch']:02d} train_loss={h['train_loss']:.4f} "
            f"val_loss={h['val_loss']:.4f} P={h['val_precision']:.3f} "
            f"R={h['val_recall']:.3f} F1={h['val_f1']:.3f} AUC={h['val_auc']:.3f}"
            for h in history
        )
        + f"\n\ntest={json.dumps(test)}\n",
        encoding="utf-8",
    )
    print("Saved", pt_deploy, onnx_path, out_json)


if __name__ == "__main__":
    main()
