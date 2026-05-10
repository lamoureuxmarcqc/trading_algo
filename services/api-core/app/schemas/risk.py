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


class ScenarioResult(BaseModel):
    scenario_id: str
    name: str
    estimated_pnl_impact: float
    drawdown_impact: float
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

