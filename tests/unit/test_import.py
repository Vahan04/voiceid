"""Basic import tests."""

import voiceid


def test_version_exists() -> None:
    """Package should expose version."""
    assert isinstance(voiceid.__version__, str)
    assert len(voiceid.__version__) > 0