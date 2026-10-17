"""Tests du registry."""
import pytest

from app.registry import RegistryError, list_models, promote, register, rollback


@pytest.fixture(autouse=True)
def clear_store():
    from app import registry
    registry._models.clear()
    yield
    registry._models.clear()


def test_register_returns_id():
    m = register({"name": "lstm", "version": "v1", "base_model": "llama"})
    assert "id" in m
    assert m["status"] == "staging"


def test_register_requires_name_and_version():
    with pytest.raises(RegistryError):
        register({"name": "", "version": "v1"})


def test_promote_requires_metrics():
    m = register({"name": "lstm", "version": "v1", "base_model": "x"})
    with pytest.raises(RegistryError):
        promote(m["id"])


def test_promote_with_metrics():
    m = register({
        "name": "lstm", "version": "v1", "base_model": "x",
        "metrics": {"factuality": 0.9},
    })
    result = promote(m["id"])
    assert result["status"] == "production"


def test_rollback_archives():
    m = register({
        "name": "lstm", "version": "v1", "base_model": "x",
        "metrics": {"factuality": 0.9},
    })
    promote(m["id"])
    result = rollback(m["id"])
    assert result["status"] == "archived"


def test_list_models():
    register({"name": "a", "version": "v1", "base_model": "x"})
    register({"name": "b", "version": "v1", "base_model": "x"})
    assert len(list_models()) == 2
