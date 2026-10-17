"""Registry : enregistrement, promotion, rollback."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class RegistryError(Exception):
    """Erreur du registry."""


_models: dict[str, dict] = {}


def register(data: dict) -> dict:
    """Enregistre une nouvelle version de modele."""
    if not data.get("name") or not data.get("version"):
        raise RegistryError("name et version sont obligatoires")

    model_id = str(uuid.uuid4())
    record = {
        "id": model_id,
        "name": data["name"],
        "version": data["version"],
        "base_model": data.get("base_model"),
        "dataset_id": data.get("dataset_id"),
        "parameters": data.get("parameters") or {},
        "metrics": data.get("metrics") or {},
        "provenance": data.get("provenance"),
        "status": "staging",
        "created_at": _utcnow_iso(),
    }
    _models[model_id] = record
    return record


def promote(model_id: str, target: str = "production") -> dict:
    """Promeut une version en production."""
    if model_id not in _models:
        raise RegistryError(f"Modele introuvable : {model_id}")

    record = _models[model_id]
    if not record.get("metrics"):
        raise RegistryError("Impossible de promouvoir sans metriques")

    record["status"] = target
    return record


def rollback(model_id: str) -> dict:
    """Rollback vers archived."""
    if model_id not in _models:
        raise RegistryError(f"Modele introuvable : {model_id}")

    _models[model_id]["status"] = "archived"
    return _models[model_id]


def list_models() -> list[dict]:
    return list(_models.values())
