"""Custom exceptions for VoiceID."""


class VoiceIDError(Exception):
    """Base exception for all VoiceID errors."""


class ConfigurationError(VoiceIDError):
    """Raised when config is invalid."""