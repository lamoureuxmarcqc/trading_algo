from pydantic import BaseModel


class ResearchIdea(BaseModel):
    symbol: str
    sector: str
    price: float
    buy_probability: float
    expected_return: float
    confidence_score: float
    factor_score: float
    market_regime: str


class FactorRank(BaseModel):
    symbol: str
    sector: str
    momentum_score: float
    quality_score: float
    volatility_score: float
    overall_score: float


class SectorRotation(BaseModel):
    sector: str
    average_buy_probability: float
    average_expected_return: float
    average_factor_score: float
    stance: str
