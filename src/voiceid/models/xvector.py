"""TDNN/x-vector style speaker encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNBlock(nn.Module):
    """1D temporal block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class StatsPooling(nn.Module):
    """Mean+std pooling across time."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1).clamp_min(1e-5)
        return torch.cat([mean, std], dim=1)


class XVectorEncoder(nn.Module):
    """Simple x-vector encoder.

    Input:  [B, n_mels, T]
    Output: [B, emb_dim]
    """

    def __init__(self, n_mels: int = 80, emb_dim: int = 192) -> None:
        super().__init__()
        self.frame_layers = nn.Sequential(
            TDNNBlock(n_mels, 256, 5),
            TDNNBlock(256, 256, 3, dilation=2),
            TDNNBlock(256, 256, 3, dilation=3),
            TDNNBlock(256, 256, 1),
            TDNNBlock(256, 512, 1),
        )
        self.pool = StatsPooling()
        self.proj = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, emb_dim),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.frame_layers(feats)
        x = self.pool(x)
        emb = self.proj(x)
        emb = F.normalize(emb, p=2, dim=1)
        return emb
