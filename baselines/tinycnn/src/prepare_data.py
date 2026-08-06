"""
Organize CASIA raw data into angry / non_angry, then split train/val/test 8:1:1.

Usage:
  python src/prepare_data.py
  python src/prepare_data.py --casia_root test/casia --out_root dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

EMOTIONS = ("angry", "fear", "happy", "neutral", "sad", "surprise")
NON_ANGRY = ("fear", "happy", "neutral", "sad", "surprise")


def collect_wavs(casia_root: Path) -> tuple[list[Path], list[Path]]:
    angry, non_angry = [], []
    speakers = [d for d in casia_root.iterdir() if d.is_dir()]
    for spk in speakers:
        for emo in EMOTIONS:
            emo_dir = spk / emo
            if not emo_dir.is_dir():
                continue
            for wav in sorted(emo_dir.glob("*.wav")):
                if emo == "angry":
                    angry.append(wav)
                elif emo in NON_ANGRY:
                    non_angry.append(wav)
    return angry, non_angry


def _copy_split(
    files: list[Path],
    labels: list[str],
    dest_root: Path,
    split_name: str,
    prefix: str,
) -> None:
    for i, (src, label) in enumerate(zip(files, labels), start=1):
        out_dir = dest_root / split_name / label
        out_dir.mkdir(parents=True, exist_ok=True)
        # unique name: speaker_emotion_id
        spk = src.parent.parent.name
        dst = out_dir / f"{prefix}_{spk}_{src.stem}_{i:04d}.wav"
        shutil.copy2(src, dst)


def split_and_copy(
    angry: list[Path],
    non_angry: list[Path],
    out_root: Path,
    seed: int = 42,
) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    # angry: train 80%, temp 20% -> val 10%, test 10%
    a_train, a_temp = train_test_split(angry, test_size=0.2, random_state=seed)
    a_val, a_test = train_test_split(a_temp, test_size=0.5, random_state=seed)

    n_train, n_temp = train_test_split(non_angry, test_size=0.2, random_state=seed)
    n_val, n_test = train_test_split(n_temp, test_size=0.5, random_state=seed)

    splits = {
        "train": (a_train, n_train),
        "val": (a_val, n_val),
        "test": (a_test, n_test),
    }

    for split_name, (a_files, n_files) in splits.items():
        _copy_split(
            a_files,
            ["angry"] * len(a_files),
            out_root,
            split_name,
            "angry",
        )
        _copy_split(
            n_files,
            ["non_angry"] * len(n_files),
            out_root,
            split_name,
            "non",
        )
        print(
            f"{split_name:5s}: angry={len(a_files):4d}, "
            f"non_angry={len(n_files):4d}, total={len(a_files)+len(n_files)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare CASIA angry binary dataset")
    parser.add_argument(
        "--casia_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "test" / "casia"),
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "dataset"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    casia_root = Path(args.casia_root)
    out_root = Path(args.out_root)

    if not casia_root.is_dir():
        raise FileNotFoundError(f"CASIA root not found: {casia_root}")

    angry, non_angry = collect_wavs(casia_root)
    print(f"Found angry={len(angry)}, non_angry={len(non_angry)}")
    if not angry or not non_angry:
        raise RuntimeError("Empty class — check CASIA folder layout")

    split_and_copy(angry, non_angry, out_root, seed=args.seed)
    print(f"Done -> {out_root}")


if __name__ == "__main__":
    main()
