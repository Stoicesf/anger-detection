"""MFCC / dual-channel feature extraction for angry detection."""

from __future__ import annotations

import librosa
import numpy as np

SR = 16000
N_MFCC = 40
DURATION = 2.0
N_FFT = 512
HOP_LENGTH = 160  # 16000/160 = 100 frames/sec


def _load_fixed(wav_path: str, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    y, _ = librosa.load(wav_path, sr=sr, duration=duration, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_mfcc(
    wav_path: str,
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    duration: float = DURATION,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract single-channel MFCC.
    Returns shape (1, n_mfcc, time_frames), e.g. (1, 40, ~201).
    """
    if y is None:
        y = _load_fixed(wav_path, sr=sr, duration=duration)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    mean = np.mean(mfcc)
    std = np.std(mfcc) + 1e-6
    mfcc = (mfcc - mean) / std

    return mfcc[np.newaxis, :, :].astype(np.float32)


def extract_dual_channel(
    wav_path: str,
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    duration: float = DURATION,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """
    MFCC + F0 dual-channel features.
    Returns shape (2, n_mfcc, time_frames).
    """
    if y is None:
        y = _load_fixed(wav_path, sr=sr, duration=duration)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr, hop_length=HOP_LENGTH)
    f0 = np.nan_to_num(f0, nan=0.0) / 400.0
    f0 = np.clip(f0, 0, 1)

    # Align F0 length to MFCC frames
    t = mfcc.shape[1]
    if len(f0) < t:
        f0 = np.pad(f0, (0, t - len(f0)))
    else:
        f0 = f0[:t]

    f0_2d = np.repeat(f0[np.newaxis, :], n_mfcc, axis=0)
    dual = np.stack([mfcc, f0_2d], axis=0)

    for c in range(2):
        mean = np.mean(dual[c])
        std = np.std(dual[c]) + 1e-6
        dual[c] = (dual[c] - mean) / std

    return dual.astype(np.float32)
