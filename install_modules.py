#!/usr/bin/env python3
"""
install_modules.py — Installe les modules de la LAMOUREUX AI PLATFORM.

Usage :
    python install_modules.py status
    python install_modules.py scanner
    python install_modules.py all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SERVICES = ROOT / "services"
KNOWLEDGE = SERVICES / "knowledge"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  + {path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════════

def install_scanner() -> None:
    print("\n[Scanner] services/knowledge/app/scanner/")
    base = KNOWLEDGE / "app" / "scanner"

    write(base / "__init__.py", '''
"""Scanner de sources documentaires."""
''')

    write(base / "filters.py", '''
"""Filtres pour le scanner : extensions, taille, exclusions."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScanFilters:
    include_extensions: list[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".md", ".txt", ".html", ".htm"
    ])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "~$*", "*.tmp", "*.bak", ".git", "node_modules",
        "__pycache__", ".DS_Store", "Thumbs.db", "@eaDir"
    ])
    max_file_size_mb: int = 50
    min_file_size_bytes: int = 16


def normalize_ext(ext: str) -> str:
    ext = ext.lower()
    return ext if ext.startswith(".") else f".{ext}"


def is_extension_allowed(path: Path, include_extensions: list[str]) -> bool:
    allowed = {normalize_ext(e) for e in include_extensions}
    return path.suffix.lower() in allowed


def is_excluded(path: Path, exclude_patterns: list[str]) -> bool:
    name = path.name
    parts = path.parts
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def is_size_allowed(path: Path, max_mb: int, min_bytes: int) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return min_bytes <= size <= max_mb * 1024 * 1024


def should_scan(path: Path, filters: ScanFilters) -> bool:
    if not path.is_file():
        return False
    if is_excluded(path, filters.exclude_patterns):
        return False
    if not is_extension_allowed(path, filters.include_extensions):
        return False
    if not is_size_allowed(path, filters.max_file_size_mb, filters.min_file_size_bytes):
        return False
    return True
''')

    write(base / "hashing.py", '''
"""Hashing SHA256 pour deduplication."""
from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_SIZE = 65536


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
''')

    write(base / "walker.py", '''
"""Parcours recursif d'une source avec filtres."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .filters import ScanFilters, is_excluded, should_scan

logger = logging.getLogger(__name__)


@dataclass
class WalkResult:
    path: Path
    size_bytes: int
    modified_at: float


def walk_source(
    root: Path,
    filters: ScanFilters,
    max_files: int | None = None,
) -> Iterator[WalkResult]:
    if not root.exists():
        logger.warning("Racine introuvable : %s", root)
        return

    count = 0
    stack: list[Path] = [root]

    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError) as e:
            logger.warning("Acces refuse sur %s : %s", current, e)
            continue

        for entry in entries:
            try:
                if entry.is_dir():
                    if not is_excluded(entry, filters.exclude_patterns):
                        stack.append(entry)
                elif should_scan(entry, filters):
                    stat = entry.stat()
                    yield WalkResult(
                        path=entry,
                        size_bytes=stat.st_size,
                        modified_at=stat.st_mtime,
                    )
                    count += 1
                    if max_files and count >= max_files:
                        return
            except (PermissionError, OSError) as e:
                logger.warning("Erreur sur %s : %s", entry, e)
                continue


def count_files(root: Path, filters: ScanFilters) -> int:
    return sum(1 for _ in walk_source(root, filters))
''')

    write(base / "worker.py", '''
"""Worker : traite 1 fichier (extract -> chunk -> embed -> store)."""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import settings
from ..embeddings.local import embed_passages
from ..ingestion.pipeline import ingest_file
from ..models import DataClassification
from ..storage.pgvector import insert_document

logger = logging.getLogger(__name__)


def process_file(
    *,
    file_path: Path,
    tenant_id: str,
    classification: DataClassification,
) -> dict:
    logger.info("Traitement : %s", file_path)

    doc = ingest_file(
        file_path,
        chunk_target_tokens=settings.chunk_target_tokens,
        overlap_ratio=settings.chunk_overlap_ratio,
    )

    if not doc.chunks:
        raise ValueError(f"Aucun contenu extrait de {file_path}")

    contents = [c.content for c in doc.chunks]
    embeddings = embed_passages(contents, settings.embedding_model)

    chunks_payload = [
        {
            "content": c.content,
            "embedding": embeddings[i].tolist(),
            "page": c.page,
            "section": c.section,
            "chunk_index": c.chunk_index,
            "metadata": c.metadata,
        }
        for i, c in enumerate(doc.chunks)
    ]

    document_id = insert_document(
        tenant_id=tenant_id,
        classification=classification,
        filename=doc.filename,
        mime_type=doc.mime_type,
        metadata={"source_path": str(file_path)},
        chunks=chunks_payload,
    )

    return {
        "document_id": document_id,
        "chunks_count": len(doc.chunks),
        "filename": doc.filename,
    }
''')

    write(base / "api.py", '''
"""Endpoints FastAPI pour le scanner."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .filters import ScanFilters
from .walker import count_files, walk_source

router = APIRouter(prefix="/api/v1/knowledge/scan", tags=["scanner"])


class PreviewRequest(BaseModel):
    root_path: str
    include_extensions: list[str] | None = None
    max_files: int = 100


class PreviewResponse(BaseModel):
    root_path: str
    files_found: int
    files_listed: int
    sample: list[str]


@router.post("/preview", response_model=PreviewResponse)
async def preview_scan(req: PreviewRequest) -> PreviewResponse:
    root = Path(req.root_path)
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"Chemin introuvable: {req.root_path}")

    filters = ScanFilters()
    if req.include_extensions:
        filters.include_extensions = req.include_extensions

    total = count_files(root, filters)
    sample: list[str] = []
    for i, result in enumerate(walk_source(root, filters, max_files=req.max_files)):
        sample.append(str(result.path))
        if i >= req.max_files - 1:
            break

    return PreviewResponse(
        root_path=req.root_path,
        files_found=total,
        files_listed=len(sample),
        sample=sample,
    )
''')

    write(KNOWLEDGE / "tests" / "test_scanner_filters.py", '''
"""Tests des filtres du scanner."""
from pathlib import Path

from app.scanner.filters import (
    ScanFilters,
    is_excluded,
    is_extension_allowed,
    should_scan,
)


def test_extension_allowed_pdf():
    assert is_extension_allowed(Path("doc.pdf"), [".pdf", ".docx"])


def test_extension_rejected_mp4():
    assert not is_extension_allowed(Path("video.mp4"), [".pdf", ".docx"])


def test_exclude_pattern_temp():
    assert is_excluded(Path("~$doc.docx"), ["~$*"])


def test_exclude_pattern_dir():
    assert is_excluded(Path("proj/node_modules/foo.js"), ["node_modules"])


def test_should_scan_valid(tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"x" * 1000)
    assert should_scan(p, ScanFilters())


def test_should_scan_too_small(tmp_path: Path):
    p = tmp_path / "tiny.pdf"
    p.write_bytes(b"x")
    filters = ScanFilters(min_file_size_bytes=100)
    assert not should_scan(p, filters)


def test_should_scan_wrong_ext(tmp_path: Path):
    p = tmp_path / "video.mp4"
    p.write_bytes(b"x" * 1000)
    assert not should_scan(p, ScanFilters())


def test_should_scan_excluded(tmp_path: Path):
    p = tmp_path / "~$doc.pdf"
    p.write_bytes(b"x" * 1000)
    assert not should_scan(p, ScanFilters())
''')

    write(KNOWLEDGE / "tests" / "test_scanner_walker.py", '''
"""Tests du walker."""
from pathlib import Path

from app.scanner.filters import ScanFilters
from app.scanner.walker import count_files, walk_source


def _make_tree(tmp_path: Path) -> Path:
    (tmp_path / "a.pdf").write_bytes(b"x" * 1000)
    (tmp_path / "b.docx").write_bytes(b"x" * 1000)
    (tmp_path / "c.mp4").write_bytes(b"x" * 1000)
    (tmp_path / "~$skip.docx").write_bytes(b"x" * 1000)

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.pdf").write_bytes(b"x" * 1000)

    node = tmp_path / "node_modules"
    node.mkdir()
    (node / "e.pdf").write_bytes(b"x" * 1000)

    return tmp_path


def test_walk_finds_valid_files(tmp_path: Path):
    root = _make_tree(tmp_path)
    names = sorted(r.path.name for r in walk_source(root, ScanFilters()))
    assert "a.pdf" in names
    assert "b.docx" in names
    assert "d.pdf" in names


def test_walk_skips_invalid(tmp_path: Path):
    root = _make_tree(tmp_path)
    names = [r.path.name for r in walk_source(root, ScanFilters())]
    assert "c.mp4" not in names
    assert "~$skip.docx" not in names
    assert "e.pdf" not in names


def test_walk_max_files(tmp_path: Path):
    root = _make_tree(tmp_path)
    assert len(list(walk_source(root, ScanFilters(), max_files=1))) == 1


def test_count_files(tmp_path: Path):
    root = _make_tree(tmp_path)
    assert count_files(root, ScanFilters()) == 3


def test_walk_missing_root(tmp_path: Path):
    assert list(walk_source(tmp_path / "nope", ScanFilters())) == []
''')

    print("  Scanner OK")


# ═══════════════════════════════════════════════════════════════════
# MEMORY
# ═══════════════════════════════════════════════════════════════════

def install_memory() -> None:
    print("\n[Memory] services/memory/")
    base = SERVICES / "memory"

    write(base / "app" / "__init__.py", '''
"""Memory Layer — 4 types de memoire."""
__version__ = "0.1.0"
''')

    write(base / "app" / "config.py", '''
"""Configuration du service memory."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEM_", env_file=".env")

    database_url: str = "postgresql+psycopg://lamoureux:changeme@localhost:5433/lamoureux"
    redis_url: str = "redis://localhost:6379/1"
    short_term_ttl_seconds: int = 3600
    log_level: str = "INFO"


settings = Settings()
''')

    write(base / "app" / "models.py", '''
"""Contrats Pydantic du service memory."""
from enum import Enum
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    SHORT_TERM = "SHORT_TERM"
    LONG_TERM = "LONG_TERM"
    KNOWLEDGE = "KNOWLEDGE"
    DECISION = "DECISION"


class ShortTermAppend(BaseModel):
    tenant_id: str
    session_id: str
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class LongTermValidate(BaseModel):
    tenant_id: str
    fact: str
    source: str | None = None
    human_validation: bool = Field(..., description="Doit etre true")


class DecisionRecord(BaseModel):
    tenant_id: str
    decision: str
    reason: str
    evidence: dict | None = None
    action: str | None = None
    model_version: str | None = None
    confidence: float | None = None
    correlation_id: str | None = None


class DecisionEvaluate(BaseModel):
    outcome: dict
    evaluation: dict | None = None
''')

    write(base / "app" / "short_term.py", '''
"""Short-term memory : Redis, TTL 1h."""
from __future__ import annotations

import json
import logging

import redis

from .config import settings
from .models import ShortTermAppend

logger = logging.getLogger(__name__)

_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(settings.redis_url, decode_responses=True)
    return _client


def _key(tenant_id: str, session_id: str) -> str:
    return f"mem:short:{tenant_id}:{session_id}"


def append(payload: ShortTermAppend) -> int:
    """Ajoute un message au buffer. Retourne la taille du buffer."""
    client = _get_client()
    key = _key(payload.tenant_id, payload.session_id)
    entry = json.dumps({
        "role": payload.role,
        "content": payload.content,
    })
    client.rpush(key, entry)
    client.expire(key, settings.short_term_ttl_seconds)
    return client.llen(key)


def get_buffer(tenant_id: str, session_id: str, limit: int = 50) -> list[dict]:
    """Recupere les derniers messages."""
    client = _get_client()
    key = _key(tenant_id, session_id)
    raw = client.lrange(key, -limit, -1)
    return [json.loads(r) for r in raw]


def clear(tenant_id: str, session_id: str) -> None:
    client = _get_client()
    client.delete(_key(tenant_id, session_id))
''')

    write(base / "app" / "decision.py", '''
"""Decision memory : append-only, immuable."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .models import DecisionEvaluate, DecisionRecord


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Base(DeclarativeBase):
    pass


class DecisionRow(Base):
    __tablename__ = "decisions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    action: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    evaluation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class DecisionImmutableError(Exception):
    """Levee quand on tente de modifier une decision."""


def record(payload: DecisionRecord) -> str:
    """Enregistre une decision (INSERT only, jamais UPDATE)."""
    return str(uuid.uuid4())
''')

    write(base / "app" / "main.py", '''
"""FastAPI app — Memory Layer."""
import logging

import structlog
from fastapi import FastAPI

from . import __version__
from .config import settings
from .models import DecisionEvaluate, DecisionRecord, LongTermValidate, ShortTermAppend

logging.basicConfig(level=settings.log_level)
logger = structlog.get_logger()

app = FastAPI(title="Lamoureux Memory Layer", version=__version__)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/short-term/append")
async def short_term_append(payload: ShortTermAppend) -> dict:
    from .short_term import append
    size = append(payload)
    return {"ok": True, "buffer_size": size}


@app.post("/v1/long-term/validate")
async def long_term_validate(payload: LongTermValidate) -> dict:
    if not payload.human_validation:
        return {"ok": False, "error": "human_validation required"}
    return {"ok": True}


@app.post("/v1/decision/record")
async def decision_record(payload: DecisionRecord) -> dict:
    from .decision import record
    decision_id = record(payload)
    return {"ok": True, "decision_id": decision_id}


@app.post("/v1/decision/{decision_id}/evaluate")
async def decision_evaluate(decision_id: str, payload: DecisionEvaluate) -> dict:
    return {"ok": True, "decision_id": decision_id}
''')

    write(base / "tests" / "__init__.py", "")

    write(base / "tests" / "test_decision_immutable.py", '''
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
''')

    write(base / "tests" / "test_short_term_models.py", '''
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
''')

    write(base / "requirements.txt", '''
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.9.0
pydantic-settings>=2.5.0
sqlalchemy>=2.0.40
alembic>=1.13.0
psycopg[binary]>=3.2.9
redis>=5.0.0
structlog>=24.4.0
pytest>=8.3.0
pytest-asyncio>=0.24.0
''')

    write(base / "pyproject.toml", '''
[project]
name = "lamoureux-memory"
version = "0.1.0"
description = "Memory layer - 4 types de memoire"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
addopts = "--basetemp=.pytest_tmp"
''')

    write(base / "conftest.py", '''
"""Rend `app` importable pour pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
''')

    print("  Memory OK")


# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def install_orchestrator() -> None:
    print("\n[Orchestrator] services/orchestrator/")
    base = SERVICES / "orchestrator"

    write(base / "app" / "__init__.py", '''
"""Orchestrator — supervisor + agents."""
__version__ = "0.1.0"
''')

    write(base / "app" / "supervisor.py", '''
"""Superviseur : routing de taches vers agents."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentManifest:
    name: str
    description: str
    allowed_tools: list[str] = field(default_factory=list)
    classification_max: str = "INTERNAL"
    targets: list[str] = field(default_factory=lambda: ["LOCAL"])


def classify_task(task_description: str) -> str:
    """Classifie une tache et retourne le nom de l'agent."""
    desc = task_description.lower()
    if any(k in desc for k in ["action", "portefeuille", "investissement", "bourse"]):
        return "finance"
    if any(k in desc for k in ["sante", "medical", "symptome", "traitement"]):
        return "health"
    return "research"


def route(task_description: str) -> str:
    """Retourne l'agent a utiliser."""
    agent = classify_task(task_description)
    logger.info("Task routed to agent=%s", agent)
    return agent
''')

    write(base / "app" / "agents" / "__init__.py", '''
"""Agents specialises."""
''')

    write(base / "app" / "agents" / "base.py", '''
"""Agent abstrait."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    name: str
    allowed_tools: list[str] = field(default_factory=list)
    classification_max: str = "INTERNAL"
    budget_tokens: int = 8000
    timeout_seconds: int = 60


class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    async def run(self, task: str, context: dict) -> dict:
        ...
''')

    write(base / "app" / "agents" / "finance.py", '''
"""Agent finance (squelette)."""
from .base import Agent, AgentConfig


class FinanceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(AgentConfig(
            name="finance",
            allowed_tools=["knowledge.query", "memory.decision.record"],
            classification_max="HIGHLY_CONFIDENTIAL",
            budget_tokens=8000,
            timeout_seconds=60,
        ))

    async def run(self, task: str, context: dict) -> dict:
        return {
            "agent": "finance",
            "task": task,
            "status": "not_implemented",
            "note": "Phase 10 — Investment Agent",
        }
''')

    write(base / "app" / "main.py", '''
"""FastAPI app — Orchestrator."""
from fastapi import FastAPI
from pydantic import BaseModel

from . import __version__
from .supervisor import route

app = FastAPI(title="Lamoureux Orchestrator", version=__version__)


class RouteRequest(BaseModel):
    task: str


class RouteResponse(BaseModel):
    agent: str
    task: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/route", response_model=RouteResponse)
async def route_task(req: RouteRequest) -> RouteResponse:
    agent = route(req.task)
    return RouteResponse(agent=agent, task=req.task)
''')

    write(base / "tests" / "__init__.py", "")

    write(base / "tests" / "test_supervisor.py", '''
"""Tests du superviseur."""
from app.supervisor import classify_task, route


def test_classify_finance():
    assert classify_task("Analyse mon portefeuille d'actions") == "finance"


def test_classify_health():
    assert classify_task("J'ai des symptomes medicaux") == "health"


def test_classify_research_default():
    assert classify_task("Trouve moi des articles") == "research"


def test_route_returns_agent():
    assert route("Investir dans AAPL") == "finance"
''')

    write(base / "pyproject.toml", '''
[project]
name = "lamoureux-orchestrator"
version = "0.1.0"
description = "Agent orchestrator"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
addopts = "--basetemp=.pytest_tmp"
''')

    write(base / "conftest.py", '''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
''')

    write(base / "requirements.txt", '''
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.9.0
pytest>=8.3.0
pytest-asyncio>=0.24.0
''')

    print("  Orchestrator OK")


# ═══════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════

def install_registry() -> None:
    print("\n[Model Registry] services/model-registry/")
    base = SERVICES / "model-registry"

    write(base / "app" / "__init__.py", '''
"""Model Registry."""
__version__ = "0.1.0"
''')

    write(base / "app" / "models.py", '''
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
''')

    write(base / "app" / "registry.py", '''
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
''')

    write(base / "app" / "main.py", '''
"""FastAPI app — Model Registry."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import __version__
from .models import ModelRegistration
from .registry import RegistryError, list_models, promote, register, rollback

app = FastAPI(title="Lamoureux Model Registry", version=__version__)


class PromoteRequest(BaseModel):
    target: str = "production"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/models/register")
async def register_model(payload: ModelRegistration) -> dict:
    try:
        return register(payload.model_dump())
    except RegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/models")
async def list_all() -> list[dict]:
    return list_models()


@app.post("/v1/models/{model_id}/promote")
async def promote_model(model_id: str, req: PromoteRequest) -> dict:
    try:
        return promote(model_id, req.target)
    except RegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/models/{model_id}/rollback")
async def rollback_model(model_id: str) -> dict:
    try:
        return rollback(model_id)
    except RegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))
''')

    write(base / "tests" / "__init__.py", "")

    write(base / "tests" / "test_registry.py", '''
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
''')

    write(base / "pyproject.toml", '''
[project]
name = "lamoureux-model-registry"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "--basetemp=.pytest_tmp"
''')

    write(base / "conftest.py", '''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
''')

    write(base / "requirements.txt", '''
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.9.0
pytest>=8.3.0
''')

    print("  Registry OK")


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def install_evaluation() -> None:
    print("\n[Evaluation] services/evaluation/")
    base = SERVICES / "evaluation"

    write(base / "app" / "__init__.py", '''
"""Evaluation framework."""
__version__ = "0.1.0"
''')

    write(base / "app" / "metrics" / "__init__.py", '''
"""Metriques d'evaluation."""
''')

    write(base / "app" / "metrics" / "retrieval.py", '''
"""Metriques de retrieval."""
from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@k : proportion de pertinents dans le top-k."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if r in relevant)
    return hits / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@k : proportion de pertinents retrouves."""
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """MRR : 1/rank du premier pertinent."""
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0
''')

    write(base / "app" / "metrics" / "calibration.py", '''
"""Calibration de confidence."""
from __future__ import annotations


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """ECE : erreur de calibration (bins de confiance)."""
    if len(confidences) != len(correct):
        raise ValueError("confidences et correct doivent avoir meme taille")
    if not confidences:
        return 0.0

    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confidences):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append(i)

    n = len(confidences)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(confidences[i] for i in bucket) / len(bucket)
        avg_acc = sum(1 for i in bucket if correct[i]) / len(bucket)
        ece += (len(bucket) / n) * abs(avg_conf - avg_acc)
    return ece
''')

    write(base / "app" / "main.py", '''
"""FastAPI app — Evaluation."""
from fastapi import FastAPI

from . import __version__

app = FastAPI(title="Lamoureux Evaluation", version=__version__)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}
''')

    write(base / "tests" / "__init__.py", "")

    write(base / "tests" / "test_metrics_retrieval.py", '''
"""Tests des metriques retrieval."""
from app.metrics.retrieval import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_precision_perfect():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_half():
    assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == 0.5


def test_recall_full():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5


def test_mrr_first_hit():
    assert mean_reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_mrr_second_hit():
    assert mean_reciprocal_rank(["x", "a"], {"a"}) == 0.5


def test_mrr_no_hit():
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0
''')

    write(base / "tests" / "test_metrics_calibration.py", '''
"""Tests calibration."""
from app.metrics.calibration import expected_calibration_error


def test_ece_perfectly_calibrated():
    confs = [0.5] * 10
    correct = [True] * 5 + [False] * 5
    ece = expected_calibration_error(confs, correct)
    assert ece < 0.01


def test_ece_overconfident():
    confs = [0.9] * 10
    correct = [True] * 5 + [False] * 5
    ece = expected_calibration_error(confs, correct)
    assert ece > 0.3


def test_ece_empty():
    assert expected_calibration_error([], []) == 0.0
''')

    write(base / "pyproject.toml", '''
[project]
name = "lamoureux-evaluation"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "--basetemp=.pytest_tmp"
''')

    write(base / "conftest.py", '''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
''')

    write(base / "requirements.txt", '''
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pytest>=8.3.0
''')

    print("  Evaluation OK")


# ═══════════════════════════════════════════════════════════════════
# SELF-IMPROVEMENT
# ═══════════════════════════════════════════════════════════════════

def install_self_improvement() -> None:
    print("\n[Self-Improvement] services/self-improvement/")
    base = SERVICES / "self-improvement"

    write(base / "app" / "__init__.py", '''
"""Self-Improvement Engine — governed pipeline."""
__version__ = "0.1.0"
''')

    write(base / "app" / "models.py", '''
"""Contrats Pydantic."""
from enum import Enum
from pydantic import BaseModel, Field


class ProposalType(str, Enum):
    CODE = "CODE"
    AI = "AI"
    KNOWLEDGE = "KNOWLEDGE"
    FEATURE = "FEATURE"
    AGENT = "AGENT"


class ProposalStatus(str, Enum):
    PENDING = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"


class Proposal(BaseModel):
    id: str
    type: ProposalType
    target: str
    issue: str
    proposal: str
    risk: str = "LOW"
    status: ProposalStatus = ProposalStatus.PENDING
    requires_human_review: bool = True
    auto_rollback_on: list[str] = Field(default_factory=list)
''')

    write(base / "app" / "detector.py", '''
"""Detector : identifie les opportunites d'amelioration."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .models import Proposal, ProposalStatus, ProposalType


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def detect_code_issues(metrics: dict) -> list[Proposal]:
    """Detecte des opportunites d'amelioration du code."""
    proposals: list[Proposal] = []
    if metrics.get("test_coverage", 1.0) < 0.7:
        proposals.append(Proposal(
            id=str(uuid.uuid4()),
            type=ProposalType.CODE,
            target="coverage",
            issue=f"Couverture de tests : {metrics['test_coverage']:.2%}",
            proposal="Ajouter des tests manquants",
            risk="LOW",
        ))
    return proposals


def detect_ai_issues(metrics: dict) -> list[Proposal]:
    """Detecte des opportunites d'amelioration de l'IA."""
    proposals: list[Proposal] = []
    if metrics.get("hallucination_rate", 0.0) > 0.15:
        proposals.append(Proposal(
            id=str(uuid.uuid4()),
            type=ProposalType.AI,
            target="factuality",
            issue=f"Hallucination rate : {metrics['hallucination_rate']:.2%}",
            proposal="Renforcer les instructions de citation dans les prompts",
            risk="MEDIUM",
            auto_rollback_on=["hallucination_rate > 0.20"],
        ))
    return proposals


def detect(metrics: dict) -> list[Proposal]:
    """Detecte toutes les opportunites selon les 5 dimensions."""
    return (
        detect_code_issues(metrics)
        + detect_ai_issues(metrics)
    )
''')

    write(base / "app" / "observer.py", '''
"""Observer : collecte des metriques multi-sources."""
from __future__ import annotations


def collect_metrics() -> dict:
    """
    Collecte les metriques depuis les autres services.

    V1 : stub. A connecter a evaluation, registry, api-core.
    """
    return {
        "test_coverage": 1.0,
        "hallucination_rate": 0.0,
        "factuality": 1.0,
    }
''')

    write(base / "app" / "main.py", '''
"""FastAPI app — Self-Improvement."""
from fastapi import FastAPI

from . import __version__
from .detector import detect
from .observer import collect_metrics

app = FastAPI(title="Lamoureux Self-Improvement", version=__version__)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/v1/improvement/proposals")
async def list_proposals() -> list[dict]:
    metrics = collect_metrics()
    proposals = detect(metrics)
    return [p.model_dump() for p in proposals]
''')

    write(base / "tests" / "__init__.py", "")

    write(base / "tests" / "test_detector.py", '''
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
''')

    write(base / "pyproject.toml", '''
[project]
name = "lamoureux-self-improvement"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "--basetemp=.pytest_tmp"
''')

    write(base / "conftest.py", '''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
''')

    write(base / "requirements.txt", '''
fastapi>=0.115.0
uvicorn[standard]==0.30.0
pydantic>=2.9.0
pytest>=8.3.0
''')

    print("  Self-Improvement OK")


# ═══════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════

def show_status() -> None:
    print("\n=== LAMOUREUX AI PLATFORM — Modules ===\n")
    checks = [
        ("ai-gateway", "services/ai-gateway/app/main.py"),
        ("api-core", "services/api-core/app/main.py"),
        ("knowledge", "services/knowledge/app/storage/models.py"),
        ("knowledge/scanner", "services/knowledge/app/scanner/walker.py"),
        ("memory", "services/memory/app/main.py"),
        ("orchestrator", "services/orchestrator/app/main.py"),
        ("model-registry", "services/model-registry/app/main.py"),
        ("evaluation", "services/evaluation/app/main.py"),
        ("self-improvement", "services/self-improvement/app/main.py"),
    ]
    for name, rel in checks:
        exists = (ROOT / rel).exists()
        icon = "OK" if exists else "--"
        print(f"  [{icon}] {name}")
    print()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

INSTALLERS = {
    "scanner": install_scanner,
    "memory": install_memory,
    "orchestrator": install_orchestrator,
    "registry": install_registry,
    "evaluation": install_evaluation,
    "self-improvement": install_self_improvement,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "module",
        choices=list(INSTALLERS.keys()) + ["all", "status"],
        default="status",
        nargs="?",
    )
    args = parser.parse_args()

    if args.module == "status":
        show_status()
        return 0

    if args.module == "all":
        for installer in INSTALLERS.values():
            installer()
        print("\nTous les modules installes.")
    else:
        INSTALLERS[args.module]()

    print()
    show_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())