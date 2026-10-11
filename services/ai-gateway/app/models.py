"""Contrats Pydantic."""
from enum import Enum
from pydantic import BaseModel, Field


class DataClassification(str, Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    HIGHLY_CONFIDENTIAL = "HIGHLY_CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


class ModelTarget(str, Enum):
    LOCAL = "LOCAL"
    LOCAL_GPU = "LOCAL_GPU"
    REMOTE = "REMOTE"
    AUTO = "AUTO"


class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    classification: DataClassification
    tenant_id: str = Field(..., min_length=1)
    model: str | None = None
    target: ModelTarget = ModelTarget.AUTO
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    system: str | None = None

class Source(BaseModel):
    document_id: str
    page: int | None = None
    excerpt: str | None = None


class InferenceResponse(BaseModel):
    content: str
    model: str
    target: ModelTarget
    tokens_in: int
    tokens_out: int
    latency_ms: int
    confidence: float | None = None
    sources: list[Source] = []


class HealthResponse(BaseModel):
    status: str
    version: str
    providers: dict[str, bool]
