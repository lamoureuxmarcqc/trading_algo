"""Tests du detector."""
from app.detector import detect, detect_ai_issues, detect_code_issues


def test_detect_code_low_coverage():
    proposals = detect_code_issues({"test_coverage": 0.5})
    assert len(proposals) == 1
    assert proposals[0].type.value == "CODE"


def test_detect_code_ok_coverage():
    proposals = detect_code_issues({"test_coverage": 0.9})
    assert proposals == []


def test_detect_ai_high_hallucination():
    proposals = detect_ai_issues({"hallucination_rate": 0.20})
    assert len(proposals) == 1
    assert proposals[0].type.value == "AI"


def test_detect_ai_ok():
    proposals = detect_ai_issues({"hallucination_rate": 0.05})
    assert proposals == []


def test_detect_combined():
    proposals = detect({"test_coverage": 0.5, "hallucination_rate": 0.20})
    assert len(proposals) == 2


def test_proposals_require_review():
    proposals = detect({"test_coverage": 0.5})
    assert proposals[0].requires_human_review is True
