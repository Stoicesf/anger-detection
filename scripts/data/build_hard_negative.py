"""Build Chinese hard-negative set for low false-alarm fine-tuning.

Sources (Chinese only):
  - CASIA: happy / surprise / fear (high-arousal non-anger) + augmented
  - CNEV: happiness (excited non-anger)

Output layout:
  data/curated/hard_negative/
    loud_normal/   # gain-boosted neutral
    excited/       # happy / happiness
    surprise/      # surprise
    fear/          # fear (non-anger)
    speech/        # speed-perturbed neutral/happy
    manifest.json

Usage:
  python scripts/data/build_hard_negative.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from anger_detection.common import CASIA_ROOT, CNEV_ROOT, CURATED_DIR, PROJECT_ROOT

CASIA = CASIA_ROOT
CNEV = CNEV_ROOT
OUT = CURATED_DIR / "hard_negative"
SR = 16000


def load(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def save(path: Path, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = np.clip(y / peak * 0.95, -1, 1)
    sf.write(str(path), y, SR)


def add_noise(y: np.ndarray, rng: np.random.Generator, snr_db: float = 15.0) -> np.ndarray:
    noise = rng.normal(0, 1, size=y.shape).astype(np.float32)
    # pink-ish by simple filter
    noise = np.convolve(noise, np.array([1, 0.5, 0.25], dtype=np.float32), mode="same")
    p_sig = float(np.mean(y**2) + 1e-9)
    p_noi = float(np.mean(noise**2) + 1e-9)
    scale = np.sqrt(p_sig / (p_noi * 10 ** (snr_db / 10)))
    return (y + noise * scale).astype(np.float32)


def change_speed(y: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)


def collect() -> dict[str, list[Path]]:
    items: dict[str, list[Path]] = {
        "excited": [],
        "surprise": [],
        "fear": [],
        "loud_normal": [],
    }
    # CASIA
    for p in sorted((CASIA / "happy").glob("*.wav")):
        items["excited"].append(p)
    for p in sorted((CASIA / "surprise").glob("*.wav")):
        items["surprise"].append(p)
    for p in sorted((CASIA / "fear").glob("*.wav")):
        items["fear"].append(p)
    for p in sorted((CASIA / "neutral").glob("*.wav")):
        items["loud_normal"].append(p)
    # CNEV happiness
    for gender in ("female", "male"):
        d = CNEV / gender / "happiness"
        if d.is_dir():
            items["excited"].extend(sorted(d.glob("*.wav")))
        n = CNEV / gender / "neutral"
        if n.is_dir():
            items["loud_normal"].extend(sorted(n.glob("*.wav"))[:80])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    src = collect()
    manifest = []
    counts = {}

    # copy / augment into buckets
    for bucket, paths in src.items():
        rng.shuffle(paths)
        paths = paths[: args.max_per_class]
        n = 0
        for i, p in enumerate(paths):
            y = load(p)
            # original
            dst = OUT / bucket / f"{bucket}_{i:04d}_orig.wav"
            save(dst, y)
            manifest.append(
                {
                    "path": str(dst.relative_to(PROJECT_ROOT)),
                    "bucket": bucket,
                    "source": str(p),
                    "aug": "orig",
                    "language": "zh",
                }
            )
            n += 1
            # gain boost (loud normal / excited hard case)
            if bucket in ("loud_normal", "excited") and rng.random() < 0.8:
                g = float(rng.uniform(1.6, 2.5))
                dst2 = OUT / bucket / f"{bucket}_{i:04d}_gain.wav"
                save(dst2, y * g)
                manifest.append(
                    {
                        "path": str(dst2.relative_to(PROJECT_ROOT)),
                        "bucket": bucket,
                        "source": str(p),
                        "aug": f"gain_{g:.2f}",
                        "language": "zh",
                    }
                )
                n += 1
            # noise
            if rng.random() < 0.35:
                snr = float(rng.uniform(8, 20))
                dst3 = OUT / bucket / f"{bucket}_{i:04d}_noise.wav"
                save(dst3, add_noise(y, rng, snr))
                manifest.append(
                    {
                        "path": str(dst3.relative_to(PROJECT_ROOT)),
                        "bucket": bucket,
                        "source": str(p),
                        "aug": f"noise_snr{snr:.1f}",
                        "language": "zh",
                    }
                )
                n += 1
        counts[bucket] = n

    # speech speed perturbations → speech/
    speech_n = 0
    base = src["loud_normal"][:150] + src["excited"][:150]
    for i, p in enumerate(base):
        y = load(p)
        rate = float(rng.choice([0.9, 0.95, 1.05, 1.1]))
        dst = OUT / "speech" / f"speech_{i:04d}_spd{rate:.2f}.wav"
        save(dst, change_speed(y, rate))
        manifest.append(
            {
                "path": str(dst.relative_to(PROJECT_ROOT)),
                "bucket": "speech",
                "source": str(p),
                "aug": f"speed_{rate}",
                "language": "zh",
            }
        )
        speech_n += 1
    counts["speech"] = speech_n

    meta = {
        "language": "zh-CN",
        "note": "Hard negatives for Chinese anger detector (non-anger high-energy / noisy / fast speech)",
        "counts": counts,
        "total": len(manifest),
        "label_mapping_for_finetune": {
            "loud_normal": 0,
            "speech": 0,
            "excited": 1,
            "surprise": 4,
            "fear": 0,
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps({"meta": meta, "items": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        "# Hard Negative（中文）\n\n"
        "仅含中文非愤怒困难样本，用于降低真实场景误报。\n\n"
        f"统计：{json.dumps(counts, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    print("Built", OUT)
    print("counts", counts, "total", len(manifest))


if __name__ == "__main__":
    main()
