"""Minimal trainer."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from voiceid.features.log_mel import LogMelSpectrogram
from voiceid.models.xvector import XVectorEncoder


@dataclass(slots=True)
class TrainStepOutput:
    """Single-step output."""

    loss: float


class Trainer:
    """Minimal training engine for speaker classification."""

    def __init__(
        self,
        feature_extractor: LogMelSpectrogram,
        model: XVectorEncoder,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ) -> None:
        self.feature_extractor = feature_extractor.to(device)
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> TrainStepOutput:
        self.model.train()
        wav, labels = batch
        wav = wav.to(self.device)
        labels = labels.to(self.device)

        feats = self.feature_extractor(wav)
        emb = self.model(feats)
        loss = self.loss_fn(emb, labels)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return TrainStepOutput(loss=float(loss.detach().cpu().item()))

    def train_epoch(self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> float:
        losses: list[float] = []
        for wav, label in loader:
            if wav.ndim == 2:
                wav = wav.unsqueeze(1)
            out = self.train_step((wav, label))
            losses.append(out.loss)
        return float(sum(losses) / max(len(losses), 1))
