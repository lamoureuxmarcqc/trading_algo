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
