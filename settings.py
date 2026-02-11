from dataclasses import dataclass
from typing import Optional

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
