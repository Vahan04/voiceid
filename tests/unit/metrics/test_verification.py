import numpy as np

from voiceid.metrics.verification import compute_eer


def test_compute_eer_range() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)
    labels = np.array([1, 1, 0, 0], dtype=int)
    eer = compute_eer(scores, labels)
    assert 0.0 <= eer <= 1.0
