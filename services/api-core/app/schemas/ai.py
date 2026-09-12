from pydantic import BaseModel, Field

from app.schemas.ai_base import ForecastResponse, RegimeResponse, SignalResponse
from app.schemas.trading import OrderResponse


from app.schemas.terminal import (  # noqa: E402
    TradingAlgoBacktestSummary,
    TradingAlgoFeatureContribution,
    TradingAlgoModelDiagnostics,
    TradingAlgoPositionSizing,
    TradingAlgoPredictionBand,
)


class AiStalenessFlags(BaseModel):
    model_stale: bool
    market_data_stale: bool
    data_drift_detected: bool
    concept_drift_detected: bool
    regime_mismatch: bool = False


class AiDiagnosticsResponse(BaseModel):
    symbol: str
    model_version: str
    feature_set_version: str
    trained_at: str | None
    inference_at: str
    confidence_score: float
    directional_accuracy_30d: float | None
    directional_accuracy_90d: float | None
    hit_rate: float | None
    mae: float | None
    rmse: float | None
    signal_sharpe_30d: float | None
    signal_sharpe_90d: float | None
    staleness: AiStalenessFlags
    diagnostics: TradingAlgoModelDiagnostics


class AiExplanationResponse(BaseModel):
    symbol: str
    source: str
    explained_at: str
    features: list[TradingAlgoFeatureContribution]
    fallback_labeled: bool


class AiPredictionBandsResponse(BaseModel):
    symbol: str
    as_of: str | None
    latest_price: float | None
    bands: list[TradingAlgoPredictionBand]


class AiSizingRequest(BaseModel):
    symbol: str
    nav: float = Field(default=1_000_000, gt=0)
    max_position_weight: float = Field(default=0.12, gt=0, le=1)
    period: str = "1y"


class AiSizingResponse(BaseModel):
    symbol: str
    sizing: TradingAlgoPositionSizing
    warnings: list[str] = Field(default_factory=list)


class AiBacktestRequest(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["AAPL"])
    lookback: str = "180d"
    buy_probability_threshold: float = Field(default=0.75, ge=0, le=1)
    max_position_weight: float = Field(default=0.12, gt=0, le=1)
    stop_loss: float = Field(default=0.08, gt=0, le=0.5)
    take_profit: float = Field(default=0.18, gt=0, le=1)
    slippage_bps: float = Field(default=5, ge=0, le=100)
    period: str = "1y"


class AiBacktestSymbolResult(BaseModel):
    symbol: str
    summary: TradingAlgoBacktestSummary
    warnings: list[str] = Field(default_factory=list)


class AiBacktestResponse(BaseModel):
    generated_at: str
    lookback: str
    buy_probability_threshold: float
    results: list[AiBacktestSymbolResult]
    errors: list[str] = Field(default_factory=list)


class AiPaperStageRequest(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["AAPL"])
    buy_probability_threshold: float = Field(default=0.75, ge=0, le=1)
    max_notional: float = Field(default=25_000, gt=0)
    max_position_weight: float = Field(default=0.12, gt=0, le=1)
    stop_loss: float = Field(default=0.08, gt=0, le=0.5)
    take_profit: float | None = Field(default=0.18, gt=0, le=1)
    policy_id: str = "paper-ai-guardrail-v1"
    paper_trading_enabled: bool = False
    period: str = "1y"
    nav: float = Field(default=1_000_000, gt=0)


class AiPaperStageSkip(BaseModel):
    symbol: str
    reason: str
    warnings: list[str] = Field(default_factory=list)


class AiPaperStageResponse(BaseModel):
    generated_at: str
    policy_id: str
    status: str
    replayed: bool
    orders: list[OrderResponse] = Field(default_factory=list)
    skipped: list[AiPaperStageSkip] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    audit_event: str = "com.terminal.ai.paper_order.staged.v1"
