"""Tests des contrats short-term."""
from app.models import ShortTermAppend


def test_short_term_role_validation():
    entry = ShortTermAppend(
        tenant_id="t1", session_id="s1", role="user", content="hello"
    )
    assert entry.role == "user"


def test_short_term_rejects_invalid_role():
    import pytest
    with pytest.raises(Exception):
        ShortTermAppend(
            tenant_id="t1", session_id="s1", role="invalid", content="x"
        )
