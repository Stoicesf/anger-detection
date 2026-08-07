"""Log-Mel feature extraction shared by training, eval, and demos."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from anger_detection.common import HOP_LENGTH, N_FFT, N_MELS, SR, TIME_FRAMES

CROPS_PER_FILE = 4


def log_mel_full(path: Path) -> np.ndarray:
    """16 kHz mono -> full-length (T, N_MELS) dB log-mel."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    # per-file RMS loudness normalization (RAVDESS mirror is very quiet;
    # this also aligns CREMA-D and the on-device microphone gain)
    rms = float(np.sqrt((y ** 2).mean()))
    if rms > 1e-4:
        y = y * (0.05 / rms)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    # dB-scale log-mel, clipped and mapped to [0, 1]
    db = librosa.power_to_db(mel, ref=1.0)
    db = np.clip(db, -60.0, 20.0)
    feat = (db + 60.0) / 80.0  # (N_MELS, T)
    return feat.T.astype(np.float32)  # (T, N_MELS)


def crop_patches(feat: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Return CROPS_PER_FILE fixed-size (TIME_FRAMES, N_MELS) patches."""
    t = feat.shape[0]
    patches: list[np.ndarray] = []
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
