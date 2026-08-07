"""Build Log-Mel features from English corpora -> X.npy / y.npy.

Usage:
  python scripts/data/prepare_data.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from anger_detection.common import (
    CREMAD_DIR,
    N_MELS,
    NUM_CLASSES,
    PROCESSED_DIR,
    RAVDESS_SPEECH_DIR,
    TIME_FRAMES,
    ensure_dirs,
)
from anger_detection.features import CROPS_PER_FILE, crop_patches, log_mel_full

RAVDESS_EMOTION = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
CREMAD_EMOTION = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def ravdess_label(filename: str):
    """Map a RAVDESS speech filename to one of the 5 anger-level classes."""
    parts = filename.replace(".wav", "").split("-")
    if len(parts) != 7:
        return None
    emotion = RAVDESS_EMOTION.get(parts[2])
    intensity = parts[3]
    if emotion == "neutral" or emotion == "calm":
        return 0
    if emotion == "angry":
        return 2 if intensity == "01" else 4
    return 1 if intensity == "01" else 3


def cremad_label(filename: str):
    """Map a CREMA-D filename to one of the 5 anger-level classes."""
    emotion = filename.split("_")[2]
    emotion = CREMAD_EMOTION.get(emotion)
    if emotion == "neutral":
        return 0
    if emotion == "angry":
        return 2
    return 1


def collect_files():
    rows = []
    if RAVDESS_SPEECH_DIR.exists():
        for wav in sorted(RAVDESS_SPEECH_DIR.rglob("*.wav")):
            label = ravdess_label(wav.name)
            if label is not None:
                rows.append((wav, wav.name, label, "RAVDESS"))
    if CREMAD_DIR.exists():
        for wav in sorted(CREMAD_DIR.glob("*.wav")):
            label = cremad_label(wav.name)
            if label is not None:
                rows.append((wav, wav.name, label, "CREMA-D"))
    return rows


def main() -> None:
    ensure_dirs()
    rows = collect_files()
    if not rows:
        sys.exit("No audio files found. Check data/raw layout.")

    mapping_file = PROCESSED_DIR / "label_mapping.csv"
    with open(mapping_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "dataset", "path"])
        for _, name, label, dataset in rows:
            writer.writerow([name, label, dataset, ""])
    print(f"label_mapping.csv written: {len(rows)} rows -> {mapping_file}")

    rng = np.random.default_rng(42)
    total = len(rows) * CROPS_PER_FILE
    X = np.zeros((total, TIME_FRAMES, N_MELS, 1), dtype=np.float32)
    y = np.zeros(total, dtype=np.int32)
    file_ids = np.zeros(total, dtype=np.int32)
    filenames = []
    idx = 0
    for i, (path, name, label, dataset) in enumerate(rows):
        feat = log_mel_full(path)
        for patch in crop_patches(feat, rng):
            X[idx, :, :, 0] = patch
            y[idx] = label
            file_ids[idx] = i
            filenames.append(f"{dataset}/{name}")
            idx += 1
        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(rows)}", flush=True)

    np.save(PROCESSED_DIR / "X.npy", X)
    np.save(PROCESSED_DIR / "y.npy", y)
    np.save(PROCESSED_DIR / "filenames.npy", np.array(filenames))
    np.save(PROCESSED_DIR / "file_ids.npy", file_ids)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Class distribution:", {c: int((y == c).sum()) for c in range(NUM_CLASSES)})


if __name__ == "__main__":
    main()
