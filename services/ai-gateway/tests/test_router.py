import pytest

from app.models import DataClassification, ModelTarget
from app.router import Router


def test_resolve_auto_local():
    r = Router()
    assert r.resolve(DataClassification.CONFIDENTIAL, ModelTarget.AUTO) == ModelTarget.LOCAL


def test_resolve_explicit():
    r = Router()
    assert r.resolve(DataClassification.PUBLIC, ModelTarget.REMOTE) == ModelTarget.REMOTE
