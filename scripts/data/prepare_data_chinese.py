"""Build the Chinese (Mandarin) emotion dataset: ESD zh + CSEMOTIONS -> X.npy / y.npy."""

import csv
import io
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from anger_detection.common import (
    CSEMOTIONS_DIR,
    ESD_DIR,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    NUM_CLASSES,
    PROCESSED_DIR,
    SR,
    TIME_FRAMES,
    ensure_dirs,
)


EMOTION_TO_LABEL = {
    "neutral": 0,
    "happiness": 1,
    "happy": 1,
    "anger": 2,
    "angry": 2,
    "sadness": 3,
    "sad": 3,
    "surprise": 4,
    "surprised": 4,
}

CROPS_PER_FILE = 4


def audio_to_mel(wav_bytes: bytes):
    """Decode wav bytes -> fixed-size (TIME_FRAMES, N_MELS) dB log-mel patch."""
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    rms = float(np.sqrt((data**2).mean()))
    if rms > 1e-4:
        data = data * (0.05 / rms)
    mel = librosa.feature.melspectrogram(
        y=data, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    db = librosa.power_to_db(mel, ref=1.0)
    db = np.clip(db, -60.0, 20.0)
    feat = ((db + 60.0) / 80.0).T.astype(np.float32)  # (T, 40)
    return feat


def crop_patches(feat: np.ndarray, rng: np.random.Generator):
    t = feat.shape[0]
    patches = []
    if t >= TIME_FRAMES:
        start = (t - TIME_FRAMES) // 2
        patches.append(feat[start : start + TIME_FRAMES])
        for _ in range(CROPS_PER_FILE - 1):
            s = rng.integers(0, t - TIME_FRAMES + 1)
            patches.append(feat[s : s + TIME_FRAMES])
    else:
        pad_before = (TIME_FRAMES - t) // 2
        pad_after = TIME_FRAMES - t - pad_before
        padded = np.pad(feat, ((pad_before, pad_after), (0, 0)), mode="edge")
        patches = [padded] * CROPS_PER_FILE
    return patches


def load_esd_rows():
    rows = []
    if not ESD_DIR.exists():
        print("ESD dir not found, skip:", ESD_DIR)
        return rows
    for pq in sorted(ESD_DIR.glob("train-*.parquet")):
        df = pd.read_parquet(pq)
        df = df[df["language"] == "zh"]
        for _, r in df.iterrows():
            label = EMOTION_TO_LABEL.get(str(r["emotion"]).lower())
            if label is None:
                continue
            rows.append((r["audio"]["bytes"], f"ESD_{r['speaker_id']}_{r['emotion']}_{r['transcript'][:10]}", label, "ESD"))
    print(f"ESD zh rows: {len(rows)}")
    return rows


def load_csemotions_rows():
    rows = []
    if not CSEMOTIONS_DIR.exists():
        print("CSEMOTIONS dir not found, skip:", CSEMOTIONS_DIR)
        return rows
    for pq in sorted(CSEMOTIONS_DIR.glob("*.parquet")):
        df = pd.read_parquet(pq)
        for _, r in df.iterrows():
            label = EMOTION_TO_LABEL.get(str(r["emotion"]).lower())
            if label is None:
                continue
            rows.append((r["audio"]["bytes"], f"CSEM_{r['speaker']}_{r['emotion']}_{r['text'][:10]}", label, "CSEMOTIONS"))
    print(f"CSEMOTIONS rows: {len(rows)}")
    return rows


def main() -> None:
    ensure_dirs()
    rows = load_esd_rows() + load_csemotions_rows()
    if not rows:
        sys.exit("No Chinese audio rows found - run download_chinese_data.py first")

    mapping_file = PROCESSED_DIR / "label_mapping_chinese.csv"
    with open(mapping_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "dataset"])
        for _, name, label, dataset in rows:
            writer.writerow([name, label, dataset])
    print(f"label_mapping_chinese.csv: {len(rows)} rows")

    rng = np.random.default_rng(42)
    total = len(rows) * CROPS_PER_FILE
    X = np.zeros((total, TIME_FRAMES, N_MELS, 1), dtype=np.float32)
    y = np.zeros(total, dtype=np.int32)
    file_ids = np.zeros(total, dtype=np.int32)
    filenames = []
    idx = 0
    for i, (wav_bytes, name, label, _) in enumerate(rows):
        feat = audio_to_mel(wav_bytes)
        for patch in crop_patches(feat, rng):
            X[idx, :, :, 0] = patch
            y[idx] = label
            file_ids[idx] = i
            filenames.append(name)
            idx += 1
        if (i + 1) % 2000 == 0:
            print(f"  processed {i + 1}/{len(rows)}", flush=True)

    np.save(PROCESSED_DIR / "X.npy", X)
    np.save(PROCESSED_DIR / "y.npy", y)
    np.save(PROCESSED_DIR / "file_ids.npy", file_ids)
    np.save(PROCESSED_DIR / "filenames.npy", np.array(filenames))
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Class distribution:", {c: int((y == c).sum()) for c in range(NUM_CLASSES)})


if __name__ == "__main__":
    main()
