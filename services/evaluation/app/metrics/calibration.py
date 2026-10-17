"""Calibration de confidence."""
from __future__ import annotations


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """ECE : erreur de calibration (bins de confiance)."""
    if len(confidences) != len(correct):
        raise ValueError("confidences et correct doivent avoir meme taille")
    if not confidences:
        return 0.0

    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confidences):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append(i)

    n = len(confidences)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(confidences[i] for i in bucket) / len(bucket)
        avg_acc = sum(1 for i in bucket if correct[i]) / len(bucket)
        ece += (len(bucket) / n) * abs(avg_conf - avg_acc)
    return ece
