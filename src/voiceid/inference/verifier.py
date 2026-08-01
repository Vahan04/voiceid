"""Enrollment and verification service."""

from pathlib import Path

import torch
import torch.nn.functional as F

from voiceid.audio.io import load_audio
from voiceid.features.log_mel import LogMelSpectrogram
from voiceid.models.xvector import XVectorEncoder


class SpeakerVerifier:
    """Simple in-memory enroll/verify pipeline."""

    def __init__(
        self,
        feature_extractor: LogMelSpectrogram,
        model: XVectorEncoder,
        device: str = "cpu",
    ) -> None:
        self.feature_extractor = feature_extractor.to(device).eval()
        self.model = model.to(device).eval()
        self.device = device
        self._enrollment: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def embed_file(self, path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
        wav, _ = load_audio(path, target_sample_rate=sample_rate)
        wav = wav.to(self.device)
        feats = self.feature_extractor(wav)
        emb = self.model(feats.unsqueeze(0) if feats.ndim == 2 else feats)
        return emb.squeeze(0)

    @torch.inference_mode()
    def enroll(self, speaker_id: str, path: str | Path) -> None:
        emb = self.embed_file(path)
        self._enrollment[speaker_id] = emb

    @torch.inference_mode()
    def verify(self, claimed_id: str, path: str | Path, threshold: float = 0.5) -> tuple[float, bool]:
        if claimed_id not in self._enrollment:
            raise KeyError(f"Speaker '{claimed_id}' is not enrolled.")
        probe = self.embed_file(path)
        ref = self._enrollment[claimed_id]
        score = float(F.cosine_similarity(probe.unsqueeze(0), ref.unsqueeze(0)).item())
        return score, score >= threshold
