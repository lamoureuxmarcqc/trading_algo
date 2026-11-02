from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
from typing import Any, Dict

from jose import jwt

from app.core.config import settings


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload: Dict[str, Any] = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def hash_password(password: str, salt: str | None = None) -> str:
    derived_salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), derived_salt.encode("utf-8"), 600000)
    return f"{derived_salt}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        salt, expected = password_hash.split("$", maxsplit=1)
    except ValueError:
        return False
    candidate = hash_password(password, salt).split("$", maxsplit=1)[1]
    return hmac.compare_digest(candidate, expected)
