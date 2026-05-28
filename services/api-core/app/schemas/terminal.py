from pydantic import BaseModel

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
