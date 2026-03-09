"""
Centralized logging configuration for trading_algo.

Call `init_logging()` once at process startup (CLI entrypoint or service manager).
Library modules MUST NOT call `logging.basicConfig()`; they should only get a logger
via `logging.getLogger(__name__)`.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def init_logging(
    level: Optional[str] = None,
    logfile: Optional[str] = None,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> None:
    """
    Initialize root logging for the application.

    - level: optional string like "DEBUG", "INFO". When None reads LOG_LEVEL env var or defaults to INFO.
    - logfile: optional path to a rotating file handler (parent dir created automatically).
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    # Remove existing handlers to avoid duplicate logs when modules import
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    formatter = logging.Formatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(numeric_level)
    root.addHandler(console)

    if logfile:
        parent = os.path.dirname(logfile) or "."
        os.makedirs(parent, exist_ok=True)
        fh = RotatingFileHandler(logfile, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(numeric_level)
        root.addHandler(fh)

    root.setLevel(numeric_level)
