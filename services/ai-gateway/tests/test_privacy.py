"""Tests de la politique de classification."""
import pytest

from app.models import DataClassification, ModelTarget
from app.privacy import (
    PrivacyViolation,
    allowed_targets,
    enforce,
    is_allowed,
    resolve_auto,
)


def test_restricted_only_local():
    assert allowed_targets(DataClassification.RESTRICTED) == {ModelTarget.LOCAL}


def test_restricted_rejects_remote():
    with pytest.raises(PrivacyViolation):
        enforce(DataClassification.RESTRICTED, ModelTarget.REMOTE)


def test_restricted_rejects_gpu():
    with pytest.raises(PrivacyViolation):
        enforce(DataClassification.RESTRICTED, ModelTarget.LOCAL_GPU)


def test_confidential_allows_local_and_gpu():
    assert is_allowed(DataClassification.CONFIDENTIAL, ModelTarget.LOCAL)
    assert is_allowed(DataClassification.CONFIDENTIAL, ModelTarget.LOCAL_GPU)
    assert not is_allowed(DataClassification.CONFIDENTIAL, ModelTarget.REMOTE)


def test_public_allows_all():
    for t in (ModelTarget.LOCAL, ModelTarget.LOCAL_GPU, ModelTarget.REMOTE):
        assert is_allowed(DataClassification.PUBLIC, t)


def test_resolve_auto_prefers_local():
    assert resolve_auto(DataClassification.PUBLIC) == ModelTarget.LOCAL
    assert resolve_auto(DataClassification.RESTRICTED) == ModelTarget.LOCAL
