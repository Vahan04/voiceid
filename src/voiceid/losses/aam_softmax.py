"""Additive Angular Margin Softmax."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmaxLoss(nn.Module):
    """AAM-Softmax classification head + CE loss."""

    def __init__(self, emb_dim: int, num_classes: int, margin: float = 0.2, scale: float = 30.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_normal_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.weight, p=2, dim=1)
        x = F.normalize(embeddings, p=2, dim=1)
        cosine = F.linear(x, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_cosine = torch.cos(theta + self.margin)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        logits = cosine * (1.0 - one_hot) + target_cosine * one_hot
        logits = logits * self.scale
        return F.cross_entropy(logits, labels)
