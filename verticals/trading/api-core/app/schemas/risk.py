from pydantic import BaseModel


class PositionRisk(BaseModel):
    symbol: str
    beta: float
    var_95: float
    cvar_95: float
    liquidity_score: float
    concentration_weight: float


class ScenarioShock(BaseModel):
    factor: str
    shock: float
    contribution: float


class ScenarioMacroMetric(BaseModel):
    label: str
    value: str


class ScenarioImpactItem(BaseModel):
    bucket: str
    pnl_impact: float
    comment: str


class ScenarioResult(BaseModel):
    scenario_id: str
    name: str
    period: str | None = None
    trigger: str | None = None
    summary: str | None = None
    estimated_pnl_impact: float
    drawdown_impact: float
    macro_context: list[ScenarioMacroMetric] = []
    portfolio_impacts: list[ScenarioImpactItem] = []
    shocks: list[ScenarioShock]


class PortfolioRiskSnapshot(BaseModel):
    var_95: float
    cvar_95: float
    beta: float
    drawdown: float
    gross_exposure: float
    net_exposure: float
    concentration_risk: float
    correlation_risk: float


class CorrelationMatrixResponse(BaseModel):
    symbols: list[str]
    matrix: list[list[float]]
    as_of: str | None = None
    methodology: str
