"""TinyCNN for binary angry detection (~20K params)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits; apply sigmoid only at inference
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = TinyCNN(1)
    x = torch.randn(2, 1, 40, 201)
    y = m(x)
    print(f"params={count_parameters(m)}, out={y.shape}")
