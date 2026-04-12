"""
Lightweight package-level settings shim.

This module makes `import trading_algo.settings` work reliably by proxying
values from a top-level `settings.py` (if present) or falling back to safe
defaults. Use this to avoid import errors when running the package or tests.
"""
from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List
from dotenv import load_dotenv
load_dotenv()  # Charge les variables du fichier .env dans os.environ

logger = logging.getLogger(__name__)

DEBUG = True
WEB_HOST = "127.0.0.1"
WEB_PORT = 8050
MARKET_CACHE_FILE = os.path.join(os.getcwd(), "cache", "market_overview.json")
# Try to import project-level settings.py (top-level) first.
_root_settings = None
DISPLAY_MAP = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones",
        "^GSPTSE": "TSX", "^VIX": "VIX", "GC=F": "OR",
        "CADUSD=X": "CAD/USD", "BTC-USD": "BTC"}
try:
    _root_settings = importlib.import_module("settings")
    logger.debug("Using top-level settings module")
except Exception:
    # No top-level settings found — we'll use defaults below.
    logger.debug("Top-level settings module not found; using trading_algo defaults")


# Helper to read attribute from root settings with default
def _get(attr: str, default: Any) -> Any:
    if _root_settings and hasattr(_root_settings, attr):
        return getattr(_root_settings, attr)
    return default


@dataclass
class APIConfig:
    alpha_vantage: Optional[str] = None
    fmp: Optional[str] = None
    polygon: Optional[str] = None
    twitter: Optional[str] = None
    nytimes: Optional[str] = None
    fred: Optional[str] = None  # Ajout FRED

    @classmethod
    def from_env_or_root(cls) -> "APIConfig":

        if _root_settings and hasattr(_root_settings, "APIConfig"):
            try:
                root_cfg = getattr(_root_settings, "APIConfig")
                # If root defines a dataclass, attempt to instantiate/copy
                return root_cfg.from_env() if hasattr(root_cfg, "from_env") else root_cfg
            except Exception:
                pass
        return cls(
            alpha_vantage=os.getenv("ALPHA_VANTAGE_API_KEY"),
            fmp=os.getenv("FMP_API_KEY"),
            polygon=os.getenv("POLYGON_API_KEY"),
            twitter=os.getenv("TWITTER_X_BEARER"),
            nytimes=os.getenv("NY_TIMES_API_KEY"),
            fred=os.getenv("FRED_API_KEY"), # Correspond au .env
        )


# Risk display defaults (proxy to root settings if provided)
SHARPE_BANDS: List[Tuple[float, float]] = _get(
    "SHARPE_BANDS", [(-2.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
)
SHARPE_COLORS: List[str] = _get(
    "SHARPE_COLORS", ["#d9534f", "#f0ad4e", "#5cb85c", "#2b7a2b"]
)
SHARPE_LABELS: List[str] = _get(
    "SHARPE_LABELS", ["< 0 (Poor)", "0–1 (Acceptable)", "1–2 (Good)", "≥2 (Excellent)"]
)

VAR_BANDS: List[Tuple[float, float]] = _get("VAR_BANDS", [(0, 2), (2, 5), (5, 10), (10, 100)])
VAR_COLORS: List[str] = _get("VAR_COLORS", ["#2b7a2b", "#5cb85c", "#f0ad4e", "#d9534f"])
VAR_LABELS: List[str] = _get(
    "VAR_LABELS", ["≤2% (Low)", "2–5% (Moderate)", "5–10% (High)", ">10% (Very High)"]
)

# Market cache / scheduler defaults
MARKET_REFRESH_INTERVAL_MIN: int = int(_get("MARKET_REFRESH_INTERVAL_MIN", 5))
MARKET_SAMPLE_SYMBOLS: int = int(_get("MARKET_SAMPLE_SYMBOLS", 120))
MARKET_CACHE_FILE: str = _get("MARKET_CACHE_FILE", os.path.join(os.getcwd(), "cache", "market_overview.json"))
MARKET_CACHE_TTL: int = int(_get("MARKET_CACHE_TTL", 60 * 30))

# Expose API config
API = APIConfig.from_env_or_root()

__all__ = [
    "API",
    "SHARPE_BANDS",
    "SHARPE_COLORS",
    "SHARPE_LABELS",
    "VAR_BANDS",
    "VAR_COLORS",
    "VAR_LABELS",
    "MARKET_REFRESH_INTERVAL_MIN",
    "MARKET_SAMPLE_SYMBOLS",
    "MARKET_CACHE_FILE",
    "MARKET_CACHE_TTL",
]
