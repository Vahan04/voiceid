"""Tests for log-mel feature extractor."""

import torch

from voiceid.features.log_mel import LogMelConfig, LogMelSpectrogram


def test_log_mel_output_rank() -> None:
    extractor = LogMelSpectrogram()
    waveform = torch.randn(1, 16000)  # 1 sec @ 16kHz
    feats = extractor(waveform)
    assert feats.ndim == 3


def test_log_mel_mel_bins_match_config() -> None:
    cfg = LogMelConfig(n_mels=64)
    extractor = LogMelSpectrogram(cfg)
    waveform = torch.randn(1, 32000)
    feats = extractor(waveform)
    assert feats.shape[1] == 64


def test_log_mel_no_nan_or_inf() -> None:
    extractor = LogMelSpectrogram()
    waveform = torch.zeros(1, 16000)
    feats = extractor(waveform)
    assert torch.isfinite(feats).all()
