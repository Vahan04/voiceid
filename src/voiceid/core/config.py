"""Configuration primitives for VoiceID."""

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime-level configuration."""

    seed: int = 42
    device: str = "cpu"
    deterministic: bool = True