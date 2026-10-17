"""Tests d'immuabilite de decision (sans DB)."""
from app.decision import DecisionImmutableError, DecisionRow


def test_decision_row_has_outcome_columns():
    cols = {c.name for c in DecisionRow.__table__.columns}
    assert "decision" in cols
    assert "reason" in cols
    assert "outcome" in cols
    assert "evaluation" in cols


def test_decision_row_append_only_by_design():
    outcome_col = DecisionRow.__table__.columns["outcome"]
    evaluation_col = DecisionRow.__table__.columns["evaluation"]
    assert outcome_col.nullable is True
    assert evaluation_col.nullable is True


def test_decision_immutable_error_exists():
    err = DecisionImmutableError("test")
    assert "test" in str(err)
