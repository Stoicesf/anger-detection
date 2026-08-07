"""5-class DS-CNN (PyTorch) — shared backbone for v1 training / finetune."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anger_detection.common import NUM_CLASSES

SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DSBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x


class DSCNN(nn.Module):
    """Same DS-CNN structure as the Keras version (width=24)."""

    def __init__(self, width: int = 24, num_classes: int = NUM_CLASSES, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(1, width, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(width)
        self.blocks = nn.Sequential(
            DSBlock(width, width),
            DSBlock(width, width * 2),
            DSBlock(width * 2, width * 2),
            DSBlock(width * 2, width * 4),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(width * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn0(self.conv1(x)))
        x = self.blocks(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
