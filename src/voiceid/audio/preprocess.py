"""Audio preprocessing utilities."""

import torch


def normalize_waveform(waveform: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Peak-normalize waveform to [-1, 1] range."""
    peak = waveform.abs().max()
    if peak < eps:
        return waveform
    return waveform / peak


def rms_db(waveform: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute RMS loudness in dBFS."""
    rms = torch.sqrt(torch.mean(waveform**2) + eps)
    return float(20.0 * torch.log10(rms + eps))
