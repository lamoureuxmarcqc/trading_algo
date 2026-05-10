from pydantic import BaseModel


class SignalResponse(BaseModel):
    symbol: str
    buy_probability: float
    sell_probability: float
    volatility_forecast: float
    confidence_score: float
    market_regime: str


class ForecastResponse(BaseModel):
    symbol: str
    price_target: float
    expected_return: float
    confidence_interval_low: float
    confidence_interval_high: float


class RegimeResponse(BaseModel):
    regime: str
    confidence: float
    recommendation: str

