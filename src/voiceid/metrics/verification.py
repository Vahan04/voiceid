"""Speaker verification metrics."""

import numpy as np


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Equal Error Rate (EER) from scores and binary labels.

    labels: 1 for target/same-speaker, 0 for non-target.
    """
    thresholds = np.unique(scores)
    fars: list[float] = []
    frrs: list[float] = []

    pos = labels == 1
    neg = labels == 0
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int(neg.sum()), 1)

    for t in thresholds:
        pred_pos = scores >= t
        fa = np.logical_and(pred_pos, neg).sum() / n_neg
        fr = np.logical_and(~pred_pos, pos).sum() / n_pos
        fars.append(float(fa))
        frrs.append(float(fr))

    fars_arr = np.array(fars)
    frrs_arr = np.array(frrs)
    idx = int(np.argmin(np.abs(fars_arr - frrs_arr)))
    eer = (fars_arr[idx] + frrs_arr[idx]) / 2.0
    return float(eer)
