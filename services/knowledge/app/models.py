"""Contrats Pydantic du service knowledge."""
from enum import Enum

from pydantic import BaseModel, Field


class DataClassification(str, Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    HIGHLY_CONFIDENTIAL = "HIGHLY_CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


class IngestRequest(BaseModel):
    file_path: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    classification: DataClassification
    metadata: dict | None = None


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_count: int
    classification: DataClassification


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    classification_max: DataClassification | None = None


class Source(BaseModel):
    document_id: str
    chunk_id: str
    page: int | None = None
    section: str | None = None
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    query: str
    sources: list[Source]
    confidence: float | None = None


class DocumentInfo(BaseModel):
    id: str
    filename: str
    classification: DataClassification
    ingested_at: str
    chunks_count: int