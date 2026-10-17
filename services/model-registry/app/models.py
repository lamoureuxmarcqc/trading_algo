"""Contrats Pydantic."""
from enum import Enum
from pydantic import BaseModel


class ModelStatus(str, Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelRegistration(BaseModel):
    name: str
    version: str
    base_model: str
    dataset_id: str | None = None
    parameters: dict | None = None
    metrics: dict | None = None
    provenance: str | None = None
