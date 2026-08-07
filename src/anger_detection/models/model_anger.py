"""DSCNN binary anger detector (V2).

Keeps the same lightweight backbone as train_torch.DSCNN, but:
  - returns a single logit (sigmoid at inference)
  - optionally inserts SE after each DSBlock
  - can return embedding for future contrastive heads
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (~2k params for width=24 chain)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class DSCNN_Anger(nn.Module):
    """Binary anger head on DS-CNN backbone.

    Input:  (N, 1, T=98, M=40)
    Output: (N, 1) logits  (apply sigmoid for probability)
    """

    def __init__(self, width: int = 24, dropout: float = 0.15, use_se: bool = False):
        super().__init__()
        self.use_se = use_se
        self.emb_dim = width * 4
        self.conv1 = nn.Conv2d(1, width, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(width)

        chs = [width, width, width * 2, width * 2, width * 4]
        blocks: list[nn.Module] = []
        for i in range(4):
            blocks.append(DSBlock(chs[i], chs[i + 1]))
            if use_se:
                blocks.append(SEBlock(chs[i + 1]))
        self.blocks = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.emb_dim, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn0(self.conv1(x)))
        x = self.blocks(x)
        return self.gap(x).flatten(1)

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        emb = self.forward_features(x)
        emb_d = self.dropout(emb)
        logit = self.fc(emb_d)
        if return_embedding:
            return emb, logit
        return logit


class DSCNN_AngerSigmoid(nn.Module):
    """ONNX export wrapper: logit -> probability."""

    def __init__(self, backbone: DSCNN_Anger):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(x))


def load_backbone_from_5class(
    model: DSCNN_Anger, state_dict: dict, strict_blocks: bool = False
) -> tuple[list[str], list[str], list[str]]:
    """Transfer shared weights from train_torch.DSCNN (5-class).

    Returns (loaded_keys, missing_keys, unexpected_keys).
    """
    own = model.state_dict()
    transferred = []
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("fc."):
            continue
        if k in own and own[k].shape == v.shape:
            new_sd[k] = v
            transferred.append(k)
        elif not strict_blocks and k.startswith("blocks.") and not model.use_se:
            if k in own and own[k].shape == v.shape:
                new_sd[k] = v
                transferred.append(k)
    missing = model.load_state_dict(new_sd, strict=False)
    return transferred, list(missing.missing_keys), list(missing.unexpected_keys)
