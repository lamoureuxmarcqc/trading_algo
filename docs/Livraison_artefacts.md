# Livraison des artefacts les plus utiles

Je livre dans l'ordre de valeur immédiate : ce qui débloque le développement aujourd'hui.

1. `services/ai-gateway/` — scaffold complet et fonctionnel
2. `infra/docker/docker-compose.yml` — stack locale complète
3. 6 ADRs fondateurs
4. Prompts Codex phase par phase

---

# 📦 Artefact 1/4 — `services/ai-gateway/` (scaffold fonctionnel)

## `services/ai-gateway/README.md`

```markdown
# AI Gateway

Point d'entrée unique pour tous les appels LLM de la LAMOUREUX AI PLATFORM.

## Rôle

- Abstraction modèle (Ollama, vLLM, providers distants)
- Enforcement de la classification avant tout appel réseau
- Routage intelligent (coût, latence, disponibilité)
- Fallback automatique
- Métriques (tokens, coût, latence par tenant)

## Contrat

```http
POST /v1/inference
```

## Règles

| Classification | Cibles autorisées |
|---|---|
| PUBLIC | LOCAL, LOCAL_GPU, REMOTE |
| INTERNAL | LOCAL, LOCAL_GPU, REMOTE |
| CONFIDENTIAL | LOCAL, LOCAL_GPU |
| HIGHLY_CONFIDENTIAL | LOCAL, LOCAL_GPU |
| RESTRICTED | LOCAL uniquement |

La règle est appliquée **avant** tout appel réseau.

## Lancer

```bash
uvicorn app.main:app --reload --port 8001
```
```

## `services/ai-gateway/requirements.txt`

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
pydantic-settings==2.5.2
httpx==0.27.2
structlog==24.4.0
prometheus-client==0.20.0
pytest==8.3.3
pytest-asyncio==0.24.0
```

## `services/ai-gateway/app/__init__.py`

```python
"""AI Gateway — model-agnostic inference with privacy enforcement."""

__version__ = "0.1.0"
```

## `services/ai-gateway/app/config.py`

```python
"""Configuration du gateway."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIGW_", env_file=".env")

    # Providers
    ollama_url: str = "http://ollama:11434"
    ollama_default_model: str = "llama3.1:8b"

    vllm_url: str = "http://vllm:8000"
    vllm_default_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    remote_api_url: str = ""
    remote_api_key: str = ""
    remote_default_model: str = ""

    # Politique
    default_target: str = "LOCAL"
    allow_remote_for_public: bool = True
    request_timeout_seconds: float = 60.0

    # Observabilité
    log_level: str = "INFO"


settings = Settings()
```

## `services/ai-gateway/app/models.py`

```python
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
    model_used: str
    target_used: ModelTarget
    tokens_in: int
    tokens_out: int
    latency_ms: int
    confidence: float | None = None
    sources: list[Source] = []


class HealthResponse(BaseModel):
    status: str
    version: str
    providers: dict[str, bool]
```

## `services/ai-gateway/app/privacy.py`

```python
"""Enforcement de la classification avant tout appel réseau."""
from .models import DataClassification, ModelTarget


# Matrice de routage — source de vérité
ALLOWED_TARGETS: dict[DataClassification, set[ModelTarget]] = {
    DataClassification.PUBLIC: {ModelTarget.LOCAL, ModelTarget.LOCAL_GPU, ModelTarget.REMOTE},
    DataClassification.INTERNAL: {ModelTarget.LOCAL, ModelTarget.LOCAL_GPU, ModelTarget.REMOTE},
    DataClassification.CONFIDENTIAL: {ModelTarget.LOCAL, ModelTarget.LOCAL_GPU},
    DataClassification.HIGHLY_CONFIDENTIAL: {ModelTarget.LOCAL, ModelTarget.LOCAL_GPU},
    DataClassification.RESTRICTED: {ModelTarget.LOCAL},
}


class PrivacyViolation(Exception):
    """Levée quand une requête viole la politique de classification."""

    def __init__(self, classification: DataClassification, requested: ModelTarget):
        self.classification = classification
        self.requested = requested
        super().__init__(
            f"Classification {classification.value} interdit la cible {requested.value}"
        )


def allowed_targets(classification: DataClassification) -> set[ModelTarget]:
    return ALLOWED_TARGETS[classification]


def is_allowed(classification: DataClassification, target: ModelTarget) -> bool:
    if target == ModelTarget.AUTO:
        return True
    return target in ALLOWED_TARGETS[classification]


def enforce(classification: DataClassification, target: ModelTarget) -> None:
    """Lève PrivacyViolation si la cible n'est pas autorisée."""
    if not is_allowed(classification, target):
        raise PrivacyViolation(classification, target)


def resolve_auto(classification: DataClassification) -> ModelTarget:
    """Choisit une cible par défaut quand AUTO est demandé."""
    targets = ALLOWED_TARGETS[classification]
    # Préférence : LOCAL > LOCAL_GPU > REMOTE
    for preferred in (ModelTarget.LOCAL, ModelTarget.LOCAL_GPU, ModelTarget.REMOTE):
        if preferred in targets:
            return preferred
    raise PrivacyViolation(classification, ModelTarget.AUTO)
```

## `services/ai-gateway/app/providers/__init__.py`

```python
from .base import Provider, ProviderError
from .ollama import OllamaProvider
from .vllm import VLLMProvider
from .remote import RemoteProvider

__all__ = ["Provider", "ProviderError", "OllamaProvider", "VLLMProvider", "RemoteProvider"]
```

## `services/ai-gateway/app/providers/base.py`

```python
"""Interface provider."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderError(Exception):
    """Erreur générique d'un provider."""


@dataclass
class ProviderResult:
    content: str
    model: str
    tokens_in: int
    tokens_out: int


class Provider(ABC):
    name: str

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str | None = None,
    ) -> ProviderResult:
        ...

    @abstractmethod
    async def health(self) -> bool:
        ...
```

## `services/ai-gateway/app/providers/ollama.py`

```python
"""Provider Ollama — inférence locale."""
import httpx
from .base import Provider, ProviderError, ProviderResult


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str, default_model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str | None = None,
    ) -> ProviderResult:
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise ProviderError(f"ollama error: {e}") from e

        return ProviderResult(
            content=data.get("response", ""),
            model=data.get("model", payload["model"]),
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
```

## `services/ai-gateway/app/providers/vllm.py`

```python
"""Provider vLLM — inférence GPU locale (API OpenAI-compatible)."""
import httpx
from .base import Provider, ProviderError, ProviderResult


class VLLMProvider(Provider):
    name = "vllm"

    def __init__(self, base_url: str, default_model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str | None = None,
    ) -> ProviderResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise ProviderError(f"vllm error: {e}") from e

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ProviderResult(
            content=choice["message"]["content"],
            model=data.get("model", payload["model"]),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
```

## `services/ai-gateway/app/providers/remote.py`

```python
"""Provider distant — OpenAI-compatible."""
import httpx
from .base import Provider, ProviderError, ProviderResult


class RemoteProvider(Provider):
    name = "remote"

    def __init__(self, base_url: str, api_key: str, default_model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str | None = None,
    ) -> ProviderResult:
        if not self.base_url or not self.api_key:
            raise ProviderError("remote provider not configured")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise ProviderError(f"remote error: {e}") from e

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ProviderResult(
            content=choice["message"]["content"],
            model=data.get("model", payload["model"]),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        )

    async def health(self) -> bool:
        return bool(self.base_url and self.api_key)
```

## `services/ai-gateway/app/router.py`

```python
"""Routage : classification → provider, avec fallback."""
import structlog
from .config import settings
from .models import DataClassification, ModelTarget
from .privacy import resolve_auto
from .providers import OllamaProvider, VLLMProvider, RemoteProvider, Provider, ProviderError

logger = structlog.get_logger()


class Router:
    def __init__(self) -> None:
        self._providers: dict[ModelTarget, Provider] = {
            ModelTarget.LOCAL: OllamaProvider(
                settings.ollama_url, settings.ollama_default_model, settings.request_timeout_seconds
            ),
            ModelTarget.LOCAL_GPU: VLLMProvider(
                settings.vllm_url, settings.vllm_default_model, settings.request_timeout_seconds * 2
            ),
            ModelTarget.REMOTE: RemoteProvider(
                settings.remote_api_url, settings.remote_api_key, settings.remote_default_model
            ),
        }

    def providers(self) -> dict[ModelTarget, Provider]:
        return self._providers

    def resolve(self, classification: DataClassification, requested: ModelTarget) -> ModelTarget:
        if requested == ModelTarget.AUTO:
            return resolve_auto(classification)
        return requested

    def get(self, target: ModelTarget) -> Provider:
        provider = self._providers.get(target)
        if provider is None:
            raise ProviderError(f"no provider for target {target.value}")
        return provider

    async def health(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for target, provider in self._providers.items():
            try:
                result[target.value] = await provider.health()
            except Exception:  # pragma: no cover - defensive
                result[target.value] = False
        return result


router = Router()
```

## `services/ai-gateway/app/main.py`

```python
"""FastAPI app — AI Gateway."""
import time
import uuid

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .models import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ModelTarget,
)
from .privacy import PrivacyViolation, enforce
from .providers import ProviderError
from .router import router

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(settings.log_level),
)

logger = structlog.get_logger()

app = FastAPI(title="Lamoureux AI Gateway", version=__version__)


@app.middleware("http")
async def correlation_id(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = cid
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    return response


@app.exception_handler(PrivacyViolation)
async def privacy_handler(request: Request, exc: PrivacyViolation):
    logger.warning(
        "privacy_violation",
        classification=exc.classification.value,
        requested=exc.requested.value,
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    return JSONResponse(
        status_code=403,
        content={
            "error": "privacy_violation",
            "detail": str(exc),
            "classification": exc.classification.value,
            "requested_target": exc.requested.value,
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    providers = await router.health()
    return HealthResponse(
        status="ok" if any(providers.values()) else "degraded",
        version=__version__,
        providers=providers,
    )


@app.post("/v1/inference", response_model=InferenceResponse)
async def inference(req: InferenceRequest, request: Request) -> InferenceResponse:
    # 1. Résolution de la cible
    target = router.resolve(req.classification, req.target)

    # 2. Enforcement privacy AVANT tout appel réseau
    enforce(req.classification, target)

    # 3. Dispatch
    provider = router.get(target)
    started = time.perf_counter()
    try:
        result = await provider.generate(
            prompt=req.prompt,
            model=req.model or "",
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            system=req.system,
        )
    except ProviderError as e:
        logger.error(
            "provider_error",
            provider=provider.name,
            error=str(e),
            correlation_id=getattr(request.state, "correlation_id", None),
        )
        raise HTTPException(status_code=502, detail=f"provider error: {e}")

    latency_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "inference",
        tenant_id=req.tenant_id,
        classification=req.classification.value,
        target=target.value,
        model=result.model,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        latency_ms=latency_ms,
        correlation_id=getattr(request.state, "correlation_id", None),
    )

    return InferenceResponse(
        content=result.content,
        model_used=result.model,
        target_used=target,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        latency_ms=latency_ms,
    )


@app.get("/v1/policy")
async def policy() -> dict:
    """Expose la matrice de routage pour audit."""
    from .privacy import ALLOWED_TARGETS
    return {
        cls.value: sorted(t.value for t in targets)
        for cls, targets in ALLOWED_TARGETS.items()
    }
```

## `services/ai-gateway/tests/test_privacy.py`

```python
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
```

## `services/ai-gateway/tests/test_router.py`

```python
import pytest

from app.models import DataClassification, ModelTarget
from app.router import Router


def test_resolve_auto_local():
    r = Router()
    assert r.resolve(DataClassification.CONFIDENTIAL, ModelTarget.AUTO) == ModelTarget.LOCAL


def test_resolve_explicit():
    r = Router()
    assert r.resolve(DataClassification.PUBLIC, ModelTarget.REMOTE) == ModelTarget.REMOTE
```

## `services/ai-gateway/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

## `services/ai-gateway/pyproject.toml`

```toml
[project]
name = "lamoureux-ai-gateway"
version = "0.1.0"
description = "Model-agnostic AI Gateway with privacy enforcement"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

---

# 📦 Artefact 2/4 — `infra/docker/docker-compose.yml`

## `infra/docker/docker-compose.yml`

```yaml
name: lamoureux-ai

services:
  # ─────────────────────────────── Données ───────────────────────────────
  postgres:
    image: pgvector/pgvector:pg16
    container_name: lamoureux-postgres
    environment:
      POSTGRES_DB: lamoureux
      POSTGRES_USER: lamoureux
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-changeme}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U lamoureux"]
      interval: 5s
      timeout: 3s
      retries: 10

  redis:
    image: redis:7-alpine
    container_name: lamoureux-redis
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

  qdrant:
    image: qdrant/qdrant:latest
    container_name: lamoureux-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrantdata:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ─────────────────────────────── Inférence ─────────────────────────────
  ollama:
    image: ollama/ollama:latest
    container_name: lamoureux-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollamadata:/root/.ollama
    # GPU optionnel — décommenter si NVIDIA
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]

  # ─────────────────────────────── AI Infrastructure ─────────────────────
  ai-gateway:
    build:
      context: ../../services/ai-gateway
    container_name: lamoureux-ai-gateway
    environment:
      AIGW_OLLAMA_URL: http://ollama:11434
      AIGW_OLLAMA_DEFAULT_MODEL: ${OLLAMA_MODEL:-llama3.1:8b}
      AIGW_VLLM_URL: http://vllm:8000
      AIGW_LOG_LEVEL: INFO
    ports:
      - "8001:8001"
    depends_on:
      ollama:
        condition: service_started
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"]
      interval: 15s
      timeout: 5s
      retries: 5

  # ─────────────────────────────── API Core (existant) ───────────────────
  api-core:
    build:
      context: ../..
      dockerfile: services/api-core/Dockerfile
    container_name: lamoureux-api-core
    environment:
      DATABASE_URL: postgresql+asyncpg://lamoureux:${POSTGRES_PASSWORD:-changeme}@postgres:5432/lamoureux
      REDIS_URL: redis://redis:6379/0
      AI_GATEWAY_URL: http://ai-gateway:8001
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      ai-gateway:
        condition: service_started
    profiles: ["full"]   # activé avec --profile full

volumes:
  pgdata:
  redisdata:
  qdrantdata:
  ollamadata:
```

## `infra/docker/.env.example`

```bash
POSTGRES_PASSWORD=changeme
OLLAMA_MODEL=llama3.1:8b
```

## `infra/docker/Makefile`

```makefile
.PHONY: up down logs ps pull-model health

up:
	docker compose -f docker-compose.yml up -d

full:
	docker compose -f docker-compose.yml --profile full up -d

down:
	docker compose -f docker-compose.yml down

logs:
	docker compose -f docker-compose.yml logs -f

ps:
	docker compose -f docker-compose.yml ps

pull-model:
	docker exec lamoureux-ollama ollama pull $${OLLAMA_MODEL:-llama3.1:8b}

health:
	@echo "→ AI Gateway"; curl -s http://localhost:8001/health | head -c 400; echo
	@echo "→ Qdrant";    curl -s http://localhost:6333/healthz; echo
	@echo "→ Postgres";  docker exec lamoureux-postgres pg_isready -U lamoureux

smoke:
	curl -s -X POST http://localhost:8001/v1/inference \
	  -H "Content-Type: application/json" \
	  -d '{"prompt":"Say hello in one word.","classification":"PUBLIC","tenant_id":"smoke"}' | head -c 400
```

---

# 📦 Artefact 3/4 — 6 ADRs fondateurs

## `docs/architecture/adr/README.md`

```markdown
# Architecture Decision Records

| ADR | Titre | Statut |
|---|---|---|
| [001](001-local-first.md) | Local-first par défaut | Accepted |
| [002](002-ai-gateway.md) | AI Gateway comme point d'entrée unique | Accepted |
| [003](003-data-classification.md) | Classification des données avant routage | Accepted |
| [004](004-memory-layers.md) | Mémoire multi-niveaux | Accepted |
| [005](005-agent-permissions.md) | Permissions des agents | Accepted |
| [006](006-trading-vertical-boundary.md) | Frontière infrastructure / vertical | Accepted |
```

## `docs/architecture/adr/001-local-first.md`

```markdown
# ADR-001 — Local-first par défaut

**Statut :** Accepted
**Date :** 2026-10-10
**Décideur :** Marc Lamoureux

## Contexte

La plateforme traite des données sensibles : financières, médicales, personnelles. Envoyer ces données à un provider distant crée un risque de fuite, de rétention, d'entraînement involontaire et de non-conformité.

## Décision

Toute donnée est traitée localement par défaut. Aucun appel réseau sortant n'est autorisé sans classification explicite et conforme à la politique. Le système doit fonctionner intégralement offline (dégradé mais fonctionnel).

## Conséquences

**Positives :**
- Confidentialité par construction
- Fonctionnement en environnement isolé (air-gap)
- Indépendance vis-à-vis des providers
- Coût marginal nul sur l'inférence locale

**Négatives :**
- Qualité dépendante des modèles locaux disponibles
- Besoin de GPU pour les gros modèles
- Coût matériel initial

## Alternatives rejetées

- **Cloud-first** : rejeté (risque de fuite, coût, dépendance).
- **Hybride par défaut** : rejeté (complexité, ambiguïté sur le routage).

## Règles

- Local est la cible par défaut.
- Remote est opt-in, jamais opt-out.
- `RESTRICTED` ne quitte jamais le local.
```

## `docs/architecture/adr/002-ai-gateway.md`

```markdown
# ADR-002 — AI Gateway comme point d'entrée unique

**Statut :** Accepted
**Date :** 2026-10-10

## Contexte

Sans couche d'abstraction, chaque service appellerait directement Ollama, vLLM ou un provider distant. Cela crée :
- duplication de logique de routage
- incohérence de la politique de confidentialité
- impossibilité de changer de modèle sans toucher au code applicatif
- absence de métriques centralisées

## Décision

Tous les appels LLM passent par `services/ai-gateway`. Aucun service n'appelle un provider directement.

## Conséquences

**Positives :**
- Politique de privacy appliquée en un seul endroit
- Changement de modèle sans modification applicative
- Métriques unifiées (tokens, coût, latence)
- Fallback centralisé
- Audit unifié

**Négatives :**
- Point de défaillance unique (mitigé par réplicas)
- Latence additionnelle (<5ms)

## Règles

- Aucun service n'appelle Ollama/vLLM/remote directement.
- Toute nouvelle route LLM est ajoutée au gateway, pas ailleurs.
- Le gateway expose `/health` et `/v1/policy` pour audit.
```

## `docs/architecture/adr/003-data-classification.md`

```markdown
# ADR-003 — Classification des données avant routage

**Statut :** Accepted
**Date :** 2026-10-10

## Contexte

Une donnée sensible ne doit jamais être routée vers un provider distant, même en cas d'erreur de configuration. La classification doit être une propriété explicite de la requête, pas une inférence.

## Décision

Cinq niveaux : `PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, `HIGHLY_CONFIDENTIAL`, `RESTRICTED`. Le gateway applique la matrice de routage **avant tout appel réseau**. Une violation lève `PrivacyViolation` (HTTP 403).

## Matrice

| Classification | LOCAL | LOCAL_GPU | REMOTE |
|---|---|---|---|
| PUBLIC | ✅ | ✅ | ✅ |
| INTERNAL | ✅ | ✅ | ✅ |
| CONFIDENTIAL | ✅ | ✅ | ❌ |
| HIGHLY_CONFIDENTIAL | ✅ | ✅ | ❌ |
| RESTRICTED | ✅ | ❌ | ❌ |

## Conséquences

- Toute requête doit porter une classification explicite.
- Le client ne peut pas désactiver la classification.
- Le gateway peut refuser une requête sans l'exécuter.

## Règles

- Défaut applicatif : `INTERNAL`.
- Les documents importés héritent de la classification du workspace.
- `RESTRICTED` = données médicales, credentials, secrets.
```

## `docs/architecture/adr/004-memory-layers.md`

```markdown
# ADR-004 — Mémoire multi-niveaux

**Statut :** Accepted
**Date :** 2026-10-10

## Contexte

Une mémoire unique crée trois problèmes : mélange de faits et d'hypothèses, impossibilité de tracer une décision, impossibilité de purger sélectivement.

## Décision

Quatre mémoires distinctes :

| Type | Storage | Nature |
|---|---|---|
| SHORT_TERM | Redis | Contexte conversation, TTL |
| LONG_TERM | PostgreSQL | Faits validés humainement |
| KNOWLEDGE | pgvector / Qdrant | Documents indexés |
| DECISION | PostgreSQL | Décisions + outcomes + évaluation |

## Règles

- Une décision enregistrée est **append-only**. On ajoute `outcome` et `evaluation`, jamais on ne modifie.
- Un fait passe de SHORT_TERM à LONG_TERM uniquement par validation humaine.
- KNOWLEDGE est adressable par tenant + classification.
- DECISION contient le modèle, la version de prompt, les sources et le confidence.

## Conséquences

- Traçabilité complète des décisions.
- Datasets d'apprentissage construits à partir de DECISION évaluée.
- Purge sélective possible (ex: purger SHORT_TERM sans toucher DECISION).
```

## `docs/architecture/adr/005-agent-permissions.md`

```markdown
# ADR-005 — Permissions des agents

**Statut :** Accepted
**Date :** 2026-10-10

## Contexte

Un agent qui hérite automatiquement des permissions de l'utilisateur peut faire plus de dégâts qu'un utilisateur (exécution en masse, appels réseau, accès fichiers). Le principe du moindre privilège doit être appliqué.

## Décision

Chaque agent possède une liste explicite de permissions :

- `tools` : liste blanche d'outils autorisés
- `classification_max` : niveau maximum qu'il peut traiter
- `targets` : cibles provider autorisées
- `budget_tokens` : limite par invocation
- `timeout_seconds`
- `sandbox` : oui/non

## Règles

- Aucun agent n'a accès direct au live trading.
- Aucun agent ne peut élever sa propre classification.
- Aucun agent ne peut ajouter un outil à sa liste.
- Toute demande d'élévation passe par ADR.
- Chaque invocation est auditée (agent, outils, budget consommé, résultat).

## Conséquences

- Les agents sont plus lents à créer (config explicite) mais sûrs.
- Un agent compromis ne peut pas dépasser son périmètre.
- Les permissions sont versionnées avec l'agent.
```

## `docs/architecture/adr/006-trading-vertical-boundary.md`

```markdown
# ADR-006 — Frontière infrastructure / vertical

**Statut :** Accepted
**Date :** 2026-10-10

## Contexte

Le trading est le premier vertical mais ne doit pas contaminer l'infrastructure partagée. Si `ai-gateway` contient de la logique de portefeuille, il ne pourra plus servir la santé ou la recherche.

## Décision

Séparation stricte :

- **Infrastructure** (`ai-gateway`, `knowledge`, `memory`, `orchestrator`, `model-registry`, `evaluation`, `self-improvement`) : aucune logique métier verticale.
- **Vertical** (`trading_algo`, `services/api-core`, `apps/web`) : logique métier, workflows, UI.

## Règles

- Un service d'infrastructure ne connaît pas "AAPL", "portefeuille", "ordre".
- Un service vertical ne réimplémente pas de routage LLM, de RAG ou de mémoire.
- Toute remontée d'une logique verticale vers l'infrastructure nécessite un ADR.
- Les contrats API de l'infrastructure sont stables et versionnés.

## Conséquences

- L'infrastructure peut servir santé, recherche, PME sans modification.
- Le trading peut évoluer sans casser l'infrastructure.
- Coût : rigueur architecturale.

## Alternatives rejetées

- **Monolithe par vertical** : rejeté (duplication).
- **Infrastructure spécifique au trading** : rejeté (non réutilisable).
```

---

# 📦 Artefact 4/4 — Prompts Codex phase par phase

## `docs/codex-prompts.md`

```markdown
# Codex — Prompts par phase

Chaque prompt est autonome. Copier-coller dans Codex après avoir lu `AGENTS.md`.

---

## Phase 0 — Fondations

```
Lis AGENTS.md et SELF_IMPROVEMENT.md.

Tâche : Phase 0 — Fondations.

1. Vérifier que les 6 ADRs existent dans docs/architecture/adr/ (001 à 006).
2. Vérifier la structure infra/docker/docker-compose.yml.
3. Créer les dossiers vides des services avec __init__.py et README.md :
   - services/ai-gateway
   - services/knowledge
   - services/memory
   - services/orchestrator
   - services/model-registry
   - services/evaluation
   - services/self-improvement
4. Créer un Makefile à la racine avec : up, down, logs, test, lint.
5. Créer .env.example à la racine.

Contraintes :
- Aucun secret commité.
- Chaque service a un pyproject.toml minimal.
- Pas de code métier à ce stade.

Ouvre une PR intitulée "phase-0: foundations".
```

---

## Phase 1 — AI Gateway

```
Lis AGENTS.md.

Tâche : Phase 1 — AI Gateway.

Le scaffold existe dans services/ai-gateway/. Complète-le :

1. Ajouter un provider "openai" (remote) dans app/providers/openai.py.
2. Ajouter un fichier app/fallback.py qui, sur échec d'un provider, tente la cible suivante autorisée par classification.
3. Ajouter un endpoint GET /v1/metrics exposant les Prometheus counters :
   - aigw_requests_total{target,classification}
   - aigw_tokens_in_total{target}
   - aigw_tokens_out_total{target}
   - aigw_latency_seconds{target}
4. Ajouter tests/test_inference.py avec :
   - test LOCAL classification PUBLIC → 200
   - test RESTRICTED + REMOTE → 403
   - test CONFIDENTIAL + REMOTE → 403
   - test fallback quand ollama indisponible
5. Mettre à jour services/ai-gateway/README.md.

Contraintes :
- Classification appliquée AVANT tout appel réseau.
- Tests passent sans réseau (mock providers).
- Type hints partout.

Ouvre une PR intitulée "phase-1: ai-gateway".
```

---

## Phase 2 — Knowledge Layer

```
Lis AGENTS.md.

Tâche : Phase 2 — Knowledge Layer.

Créer services/knowledge/ avec :

1. app/ingestion/extractors.py : PDF, DOCX, HTML, MD
2. app/chunking/semantic.py : chunking sémantique avec overlap 10%
3. app/embeddings/local.py : embeddings via sentence-transformers (BAAI/bge-small-en-v1.5)
4. app/storage/pgvector.py : schéma chunks + metadata + classification
5. app/retrieval/hybrid.py : vector + BM25 + RRF
6. app/retrieval/reranker.py : cross-encoder local
7. app/citations.py : attache document_id, page, excerpt à chaque réponse
8. app/main.py : endpoints POST /ingest, POST /query, GET /documents/{id}

Schéma DB (Alembic) :
- documents(id, tenant_id, classification, filename, ingested_at)
- chunks(id, document_id, content, embedding vector(384), page, section, metadata jsonb)

Contraintes :
- Classification héritée du document, jamais devinée.
- Toute réponse expose sources + confidence.
- RESTRICTED refusé si tenant mismatch.

Tests :
- ingestion PDF → chunks
- query → top_k avec citations
- classification respectée

Ouvre une PR intitulée "phase-2: knowledge-layer".
```

---

## Phase 3 — Memory Layer

```
Lis AGENTS.md.

Tâche : Phase 3 — Memory Layer.

Créer services/memory/ avec les 4 mémoires :

1. app/short_term.py — Redis, TTL 1h, conversation buffer
2. app/long_term.py — PostgreSQL, faits validés, table facts
3. app/knowledge.py — proxy vers services/knowledge (pas de stockage local)
4. app/decision.py — PostgreSQL, table decisions, APPEND-ONLY

Table decisions (Alembic) :
- id, tenant_id, created_at, decision, reason, evidence jsonb,
  action, outcome jsonb NULL, evaluation jsonb NULL,
  model_version, confidence, correlation_id

Règles :
- UPDATE sur decisions est interdit par trigger PostgreSQL.
- Seuls outcome et evaluation peuvent être ajoutés (colonnes NULL → valeur).
- Un trigger protège l'immuabilité.

Endpoints :
- POST /short-term/append
- POST /long-term/validate  (nécessite human_validation=true)
- POST /decision/record
- POST /decision/{id}/evaluate

Tests :
- tentative UPDATE sur decision → erreur
- evaluate ajoute outcome sans modifier decision

Ouvre une PR intitulée "phase-3: memory-layer".
```

---

## Phase 4 — Orchestrator

```
Lis AGENTS.md.

Tâche : Phase 4 — Orchestrator.

Créer services/orchestrator/ avec :

1. app/supervisor.py : classification de tâche → agent
2. app/agents/base.py : Agent abstrait (permissions, budget, timeout)
3. app/agents/research.py
4. app/agents/finance.py
5. app/agents/health.py
6. app/guardrails.py : budget tokens, timeout, sandbox, permissions tools
7. app/audit.py : émission d'événements vers outbox
8. app/main.py : POST /route, GET /agents, GET /audit/{id}

Chaque agent a un manifest YAML :
services/orchestrator/agents/finance.yaml
```yaml
name: finance
tools: [knowledge.query, memory.decision.record]
classification_max: HIGHLY_CONFIDENTIAL
targets: [LOCAL, LOCAL_GPU]
budget_tokens: 8000
timeout_seconds: 60
sandbox: true
```

Contraintes :
- Aucun agent ne peut élever sa classification.
- Aucun agent n'a accès au live trading.
- Toute invocation est auditée.

Ouvre une PR intitulée "phase-4: orchestrator".
```

---

## Phase 5 — Model Registry

```
Lis AGENTS.md.

Tâche : Phase 5 — Model Registry.

Créer services/model-registry/ avec :

1. Table models (Alembic) :
   - id, name, version, base_model, dataset_id, parameters jsonb,
     metrics jsonb, status, created_at, provenance
2. Endpoints :
   - POST /models/register
   - GET /models
   - GET /models/{id}
   - POST /models/{id}/promote
   - POST /models/{id}/rollback
   - GET /models/compare?a=&b=
3. app/provenance.py : trace dataset, code commit, params
4. app/comparison.py : compare deux versions sur métriques

Règles :
- Une version promue en production doit avoir metrics non vides.
- Rollback restaure la version précédente.
- Toute promotion est auditée.

Tests :
- register → promote → rollback

Ouvre une PR intitulée "phase-5: model-registry".
```

---

## Phase 6 — Evaluation

```
Lis AGENTS.md.

Tâche : Phase 6 — Evaluation.

Créer services/evaluation/ avec :

1. app/metrics/factuality.py — vérifie claims vs sources
2. app/metrics/hallucination.py — claims non supportés / total
3. app/metrics/retrieval.py — precision@k, recall@k
4. app/metrics/calibration.py — confidence calibration (ECE)
5. app/datasets/loader.py — datasets versionnés
6. app/runner.py — exécute une évaluation complète
7. Endpoints :
   - POST /evaluate
   - GET /evaluations/{id}
   - GET /metrics/summary

Dataset de référence :
services/evaluation/datasets/finance-eval-v1.jsonl
Format : {"query": "...", "expected_sources": [...], "reference_answer": "..."}

Règles :
- Aucun appel externe pour évaluer RESTRICTED.
- Les métriques sont stockées et immuables.
- Une évaluation référence toujours un dataset versionné.

Ouvre une PR intitulée "phase-6: evaluation".
```

---

## Phase 7 — Self-Improvement (Observer + Detector)

```
Lis AGENTS.md et SELF_IMPROVEMENT.md.

Tâche : Phase 7 — Self-Improvement Observer + Detector.

PRÉREQUIS : phases 5 et 6 complètes.

Créer services/self-improvement/ avec :

1. app/observer.py : collecte métriques depuis
   - evaluation (factuality, hallucination, calibration)
   - model-registry (drift)
   - api-core (latence, erreurs)
   - audit (agents, budgets)
2. app/detector.py : détecte opportunités selon 5 dimensions
   - CODE, AI, KNOWLEDGE, FEATURE, AGENT
3. app/proposer.py : génère propositions au format YAML (cf. SELF_IMPROVEMENT.md)
4. app/storage.py : table proposals (Alembic)

Endpoint :
- GET /improvement/proposals
- GET /improvement/proposals/{id}

Règles :
- Rate limiting strict (5 CODE/j, 3 AI/j, 10 KNOWLEDGE/j, etc.)
- Aucune proposition n'est auto-déployée.
- Toute proposition est stockée immuablement.

Tests :
- detect → propose → proposition stockée avec status="pending_review"

Ouvre une PR intitulée "phase-7: self-improvement-observer".
```

---

## Phase 8 — Self-Improvement (Simulator + Reviewer)

```
Lis AGENTS.md et SELF_IMPROVEMENT.md.

Tâche : Phase 8 — Simulator + Reviewer.

1. app/simulator.py : sandbox par dimension
   - CODE → run tests sur branche isolée
   - AI → A/B sur dataset de référence
   - KNOWLEDGE → dry-run ingestion
   - FEATURE → analytics simulation
   - AGENT → replay d'invocations passées
2. app/reviewer.py : workflow validation humaine
3. app/main.py : endpoints simulate / approve / reject / modify
4. Circuit breakers (app/breakers.py) :
   - 2 rollbacks consécutifs → pause dimension
   - factuality -15% → halt
   - hallucination > 15% → halt

Endpoints :
- POST /improvement/proposals/{id}/simulate
- POST /improvement/proposals/{id}/approve
- POST /improvement/proposals/{id}/reject

Règles :
- Aucune approbation sans simulation préalable.
- Double validation pour propositions touchant sécurité/risques/ordres/audit.
- Toute décision humaine est auditée.

Ouvre une PR intitulée "phase-8: self-improvement-simulator".
```

---

## Phase 9 — Self-Improvement (Deployer + Evaluator)

```
Lis AGENTS.md et SELF_IMPROVEMENT.md.

Tâche : Phase 9 — Deployer + Evaluator.

1. app/deployer.py : staged rollout
   - apply change sur branche/cible
   - monitoring 24h
   - promotion automatique si métriques stables
2. app/evaluator.py : métriques post-déploiement
   - factuality, hallucination, latence, erreurs
3. app/rollback.py : rollback automatique si critère violé
4. Endpoints :
   - POST /improvement/proposals/{id}/deploy
   - POST /improvement/proposals/{id}/rollback
   - GET /improvement/metrics
   - GET /improvement/audit

Règles :
- Un seul déploiement à la fois.
- Rollback window 24h.
- Toute décision finale (KEEP/ROLLBACK) est auditée.
- Circuit breaker déclenche rollback automatique.

Ouvre une PR intitulée "phase-9: self-improvement-deployer".
```

---

## Phase 10 — Investment Agent

```
Lis AGENTS.md.

Tâche : Phase 10 — Investment Agent.

1. Créer services/orchestrator/app/agents/investment.py
2. Intégrer AchatVenteAction via services/api-core
3. Chaque ordre stagé crée un record Decision Memory :
   - decision (BUY/SELL/HOLD)
   - reason
   - evidence (RAG sources)
   - action
   - outcome (NULL au départ)
4. Pre-trade checks institutionnels :
   - restricted list
   - liquidité
   - turnover max
   - concentration
5. Pro-forma blotter avant staging

Contraintes :
- Aucun ordre live.
- Idempotency-Key obligatoire.
- Classification HIGHLY_CONFIDENTIAL minimum.
- Decision Memory append-only.

Ouvre une PR intitulée "phase-10: investment-agent".
```

---

## Phase 11 — Health Workspace

```
Lis AGENTS.md.

Tâche : Phase 11 — Health Workspace.

Workspace strictement isolé : tenant dédié, DB isolée logiquement.

1. services/knowledge/workspaces/health/ : config isolation
2. Ingestion documents médicaux → classification RESTRICTED obligatoire
3. Classification des niveaux de preuve :
   FACT / SOURCE / INTERPRETATION / HYPOTHESIS / UNCERTAINTY
4. Timeline des événements médicaux
5. Détection de contradictions (signalement, pas résolution)

Règles :
- Aucune donnée santé ne sort du workspace.
- RESTRICTED → LOCAL only, sans exception.
- Aucune donnée santé ne peut être utilisée pour fine-tuning sans validation.

Ouvre une PR intitulée "phase-11: health-workspace".
```

---

## Phase 12 — Distributed GPU

```
Lis AGENTS.md.

Tâche : Phase 12 — Distributed GPU.

1. infra/vllm/ : config vLLM pour GPU server
2. infra/docker/docker-compose.gpu.yml : override GPU
3. services/ai-gateway/app/providers/vllm.py : support multi-endpoints
4. services/orchestrator/app/scheduler.py : routage par charge
5. Monitoring : Prometheus + Grafana dashboards
6. Fault tolerance : fallback GPU → LOCAL si GPU indisponible

Règles :
- RESTRICTED ne va JAMAIS sur GPU distant si GPU est hors du périmètre local.
- Le fallback respecte la politique de classification.
- Le scheduler ne peut pas router vers une cible non autorisée.

Ouvre une PR intitulée "phase-12: distributed-gpu".
```
```

---

# ✅ Résumé de la livraison

| # | Artefact | Statut | Utilité immédiate |
|---|---|---|---|
| 1 | `services/ai-gateway/` complet | ✅ Fonctionnel | Débloque tout le reste |
| 2 | `infra/docker/docker-compose.yml` | ✅ Fonctionnel | Stack locale en 1 commande |
| 3 | 6 ADRs fondateurs | ✅ Complets | Gouvernance posée |
| 4 | Prompts Codex 13 phases | ✅ Prêts | Développement piloté |

## Démarrage en 3 commandes

```bash
# 1. Stack
cd infra/docker && cp .env.example .env && make up && make pull-model

# 2. Smoke test du gateway
make smoke

# 3. Tests
cd ../../services/ai-gateway && pip install -r requirements.txt && pytest
```

## Ce qui est débloqué

- **AI Gateway opérationnel** avec enforcement privacy testé
- **Stack Docker** PostgreSQL+pgvector, Redis, Qdrant, Ollama
- **6 ADRs** qui figent les décisions structurantes
- **13 prompts Codex** pour enchaîner phases 0 → 12 sans réfléchir

## Prochaine action concrète

```bash
git checkout -b phase-0/foundations
# copier les fichiers livrés
git add . && git commit -m "phase-0: foundations"
# puis prompt Codex Phase 0
```