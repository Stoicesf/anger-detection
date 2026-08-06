"""PyTorch Dataset for angry / non_angry wav files."""

from __future__ import annotations

import glob
import os
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from feature import SR, _load_fixed, extract_mfcc


class AngryDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        feature_fn: Callable = extract_mfcc,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.samples: list[tuple[str, int]] = []
        self.feature_fn = feature_fn
        self.augment = augment
        self.transform = transform
        self._audio_augment = None

        angry_files = sorted(glob.glob(os.path.join(root_dir, "angry", "*.wav")))
        for f in angry_files:
            self.samples.append((f, 1))

        non_angry_files = sorted(glob.glob(os.path.join(root_dir, "non_angry", "*.wav")))
        for f in non_angry_files:
            self.samples.append((f, 0))

        if not self.samples:
            raise FileNotFoundError(f"No wav files found under {root_dir}")

        if augment:
            try:
                import audiomentations as A

                self._audio_augment = A.Compose(
                    [
                        A.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                        A.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
                        A.Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
                    ]
                )
            except ImportError:
                print("[Warn] audiomentations not installed, skip audio augment")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        wav_path, label = self.samples[idx]

        if self._audio_augment is not None:
            y = _load_fixed(wav_path)
            y = self._audio_augment(samples=y.astype(np.float32), sample_rate=SR)
            # re-pad/truncate after time-stretch
            target_len = int(SR * 2.0)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
            features = self.feature_fn(wav_path, y=y)
        else:
            features = self.feature_fn(wav_path)

        if self.transform:
            features = self.transform(features)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

    def class_counts(self) -> tuple[int, int]:
        n_pos = sum(1 for _, y in self.samples if y == 1)
        n_neg = len(self.samples) - n_pos
        return n_pos, n_neg
