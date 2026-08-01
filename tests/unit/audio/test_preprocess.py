"""Tests for audio preprocess."""

import torch

from voiceid.audio.preprocess import normalize_waveform, rms_db


def test_normalize_waveform_peak_is_one() -> None:
    waveform = torch.tensor([[0.2, -0.4, 0.8]], dtype=torch.float32)
    out = normalize_waveform(waveform)
    assert torch.isclose(out.abs().max(), torch.tensor(1.0))


def test_normalize_waveform_zero_safe() -> None:
    waveform = torch.zeros(1, 10)
    out = normalize_waveform(waveform)
    assert torch.equal(out, waveform)


def test_rms_db_returns_float() -> None:
    waveform = torch.ones(1, 160) * 0.5
    value = rms_db(waveform)
    assert isinstance(value, float)
