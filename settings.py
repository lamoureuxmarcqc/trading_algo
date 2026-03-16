from typing import Optional
import os
from dataclasses import dataclass

@dataclass
class APIConfig:
    alpha_vantage: Optional[str] = None
    fmp: Optional[str] = None
    polygon: Optional[str] = None
    twitter: Optional[str] = None
    nytimes: Optional[str] = None
    
    @classmethod
    def from_env(cls):
        """Charge la configuration depuis les variables d'environnement"""
        return cls(
            alpha_vantage=os.getenv('ALPHA_VANTAGE_API_KEY'),
            fmp=os.getenv('FMP_API_KEY'),
            polygon=os.getenv('POLYGON_API_KEY'),
            twitter=os.getenv('TWITTER_X_BEARER'),
            nytimes=os.getenv('NY_TIMES_API_KEY')
        )

# --- Risk display thresholds and colors (tweakable) ---
SHARPE_BANDS = [(-2, 0), (0, 1), (1, 2), (2, 3)]
SHARPE_COLORS = ["#d9534f", "#f0ad4e", "#5cb85c", "#2b7a2b"]  # red, orange, lightgreen, darkgreen
SHARPE_LABELS = ["< 0 (Poor)", "0–1 (Acceptable)", "1–2 (Good)", "≥2 (Excellent)"]

VAR_BANDS = [(0, 2), (2, 5), (5, 10), (10, 100)]
VAR_COLORS = ["#2b7a2b", "#5cb85c", "#f0ad4e", "#d9534f"]
VAR_LABELS = ["≤2% (Low)", "2–5% (Moderate)", "5–10% (High)", ">10% (Very High)"]

# --- Market cache / scheduler settings ---
# interval in minutes for background refresh of sector/top movers and indices
MARKET_REFRESH_INTERVAL_MIN = int(os.getenv("MARKET_REFRESH_INTERVAL_MIN", "10"))
# maximum symbols sampled from S&P500 for computing sector/top movers (keeps scheduled job bounded)
MARKET_SAMPLE_SYMBOLS = int(os.getenv("MARKET_SAMPLE_SYMBOLS", "120"))
# path for file-based market cache (used by Dash callbacks to avoid heavy per-request fetching)
MARKET_CACHE_FILE = os.path.join(os.getcwd(), "cache", "market_overview.json")
# TTL in seconds for market cache consumers (not enforced by writer; used by consumers as needed)
MARKET_CACHE_TTL = int(os.getenv("MARKET_CACHE_TTL", str(60 * 30)))  # default 30 minutes
