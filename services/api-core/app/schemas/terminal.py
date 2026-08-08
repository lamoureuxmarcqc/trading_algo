from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.admin import AdminUser, AuditLogEntry, DomainEventSummary
from app.schemas.ai import ForecastResponse, RegimeResponse, SignalResponse
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
