"""Audio I/O helpers."""

from pathlib import Path
from typing import Tuple

import torch
import torchaudio


def load_audio(path: str | Path, target_sample_rate: int | None = None) -> Tuple[torch.Tensor, int]:
    """Load audio from disk.

    Returns:
        waveform: Tensor [channels, time]
        sample_rate: Sampling rate in Hz
    """
    audio_path = Path(path)
    waveform, sample_rate = torchaudio.load(str(audio_path))

    if target_sample_rate is not None and sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=target_sample_rate
        )
        sample_rate = target_sample_rate

    return waveform, sample_rate


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save audio tensor to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), waveform, sample_rate)
