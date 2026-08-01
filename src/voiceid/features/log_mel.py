"""Log-Mel feature extraction."""

from dataclasses import dataclass

import torch
import torchaudio


@dataclass(slots=True)
class LogMelConfig:
    """Configuration for Log-Mel extraction."""

    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 7600.0
    clamp_min: float = 1e-5


class LogMelSpectrogram(torch.nn.Module):
    """Compute log-mel spectrograms from waveform."""

    def __init__(self, config: LogMelConfig | None = None) -> None:
        super().__init__()
        self.config = config or LogMelConfig()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="htk",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return log-mel features with shape [channel, n_mels, frames]."""
        mel_spec = self.mel(waveform)
        log_mel = torch.log(mel_spec.clamp_min(self.config.clamp_min))
        return log_mel
