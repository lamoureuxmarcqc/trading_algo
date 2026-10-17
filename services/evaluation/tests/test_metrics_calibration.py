"""Tests calibration."""
from app.metrics.calibration import expected_calibration_error


def test_ece_perfectly_calibrated():
    confs = [0.5] * 10
    correct = [True] * 5 + [False] * 5
    ece = expected_calibration_error(confs, correct)
    assert ece < 0.01


def test_ece_overconfident():
    confs = [0.9] * 10
    correct = [True] * 5 + [False] * 5
    ece = expected_calibration_error(confs, correct)
    assert ece > 0.3


def test_ece_empty():
    assert expected_calibration_error([], []) == 0.0
