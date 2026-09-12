from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.admin import AdminUser, AuditLogEntry, DomainEventSummary
from app.schemas.ai_base import ForecastResponse, RegimeResponse, SignalResponse
from app.schemas.portfolio import (
    BarbellAllocationResponse,
    PortfolioHistoryPoint,
    PortfolioOverview,
    PortfolioPerformance,
)
from app.schemas.research import FactorRank, ResearchIdea, SectorRotation
from app.schemas.risk import CorrelationMatrixResponse, PortfolioRiskSnapshot, PositionRisk, ScenarioResult
from app.schemas.trading import Fill, OrderResponse


class TerminalSnapshotResponse(BaseModel):
    generated_at: str
    portfolio: PortfolioOverview
    performance: PortfolioPerformance
    risk: PortfolioRiskSnapshot
    regime: RegimeResponse
    history: list[PortfolioHistoryPoint]
    signals: list[SignalResponse]
    forecasts: list[ForecastResponse]
    scenarios: list[ScenarioResult]
    barbell: BarbellAllocationResponse
    correlation_matrix: CorrelationMatrixResponse
    position_risk: list[PositionRisk]
    orders: list[OrderResponse]
    fills: list[Fill]
    research: list[ResearchIdea]
    factors: list[FactorRank]
    sectors: list[SectorRotation]
    users: list[AdminUser]
    audit_logs: list[AuditLogEntry]
    event_summary: DomainEventSummary


class TradingAlgoCommandRequest(BaseModel):
    command: str = "analyze"
    symbols: list[str] = Field(default_factory=lambda: ["AAPL"])
    period: str = "1y"
    max_symbols: int = 8


class TradingAlgoFeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: str
    explanation: str


class TradingAlgoPredictionInterval(BaseModel):
    confidence: float
    low: float
    high: float


class TradingAlgoPredictionBand(BaseModel):
    horizon_days: int
    median_target: float
    expected_return: float
    target_price: float | None = None
    stop_price: float | None = None
    probability_of_target: float | None = None
    probability_of_stop: float | None = None
    intervals: list[TradingAlgoPredictionInterval]


class TradingAlgoModelDiagnostics(BaseModel):
    model_version: str
    feature_set_version: str
    trained_at: Optional[str]
    inference_at: str
    confidence_score: float
    directional_accuracy_30d: Optional[float]
    directional_accuracy_90d: Optional[float]
    hit_rate: Optional[float]
    mae: Optional[float]
    rmse: Optional[float]
    signal_sharpe_30d: Optional[float]
    signal_sharpe_90d: Optional[float]
    model_stale: bool
    data_drift_detected: bool
    concept_drift_detected: bool
    market_data_stale: bool
    regime_mismatch: bool = False


class TradingAlgoPositionSizing(BaseModel):
    buy_probability: float
    model_edge: float
    kelly_fraction: float
    capped_kelly_fraction: float
    volatility_adjusted_weight: float
    recommended_weight: float
    recommended_notional: float
    recommended_quantity: float | None = None
    risk_budget_consumed: float | None = None
    stop_loss: float
    take_profit: float


class TradingAlgoBacktestSummary(BaseModel):
    lookback_days: int
    pnl: float
    annualized_return: float | None = None
    max_drawdown: float
    win_loss_ratio: float
    trade_count: int
    exposure_time: float
    average_trade: float | None = None
    fees: float | None = None
    slippage: float | None = None
    benchmark_return: float
    excess_return: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None


class TradingAlgoSymbolAnalysis(BaseModel):
    symbol: str
    period: str
    rows: int
    as_of: Optional[str]
    latest_price: Optional[float]
    daily_return: Optional[float]
    total_return: Optional[float]
    volatility_20d: Optional[float]
    sharpe_ratio: Optional[float]
    var_95: Optional[float]
    max_drawdown: Optional[float]
    rsi: Optional[float]
    sma_20: Optional[float]
    sma_50: Optional[float]
    trend: str
    recommendation: str
    diagnostics: Optional[TradingAlgoModelDiagnostics] = None
    explanation_source: str = "heuristic_fallback"
    explained_at: Optional[str] = None
    feature_importance: list[TradingAlgoFeatureContribution] = Field(default_factory=list)
    prediction_bands: list[TradingAlgoPredictionBand] = Field(default_factory=list)
    sizing: Optional[TradingAlgoPositionSizing] = None
    backtest: Optional[TradingAlgoBacktestSummary] = None


class TradingAlgoCommandResponse(BaseModel):
    command: str
    status: str
    generated_at: str
    summary: str
    analyses: list[TradingAlgoSymbolAnalysis]
    errors: list[str] = Field(default_factory=list)


class MonteCarloTrajectory(BaseModel):
    day: int
    p5_nav: float
    p50_nav: float
    p95_nav: float


class MonteCarloSimulationRequest(BaseModel):
    n_paths: int = Field(default=1000, ge=100, le=5000)
    horizon_days: int = Field(default=252, ge=21, le=756)
    proposed_weights: dict[str, float] | None = None


class MonteCarloSimulationResponse(BaseModel):
    generated_at: str
    n_paths: int
    horizon_days: int
    nav: float
    expected_annual_return: float
    simulated_sharpe_ratio: float
    var_95: float
    cvar_95: float
    trajectory: list[MonteCarloTrajectory]
    symbols: list[str]
    methodology: str
