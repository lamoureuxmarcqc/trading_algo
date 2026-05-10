from pydantic import BaseModel, Field


class PortfolioPosition(BaseModel):
    symbol: str
    quantity: float
    average_cost: float
    market_price: float
    market_value: float
    daily_pnl: float
    unrealized_pnl: float
    currency: str = "USD"


class PortfolioOverview(BaseModel):
    portfolio_id: str
    nav: float
    cash: float
    gross_exposure: float
    net_exposure: float
    base_currency: str = "USD"
    benchmark: str = "SPY"
    market_data_as_of: str | None = None
    positions: list[PortfolioPosition]


class PortfolioPerformance(BaseModel):
    day_return: float
    month_return: float
    year_return: float
    alpha_vs_benchmark: float
    sharpe_ratio: float
    max_drawdown: float


class RebalanceTarget(BaseModel):
    symbol: str
    target_weight: float = Field(ge=0, le=1)


class RebalanceRequest(BaseModel):
    targets: list[RebalanceTarget]


class RebalanceInstruction(BaseModel):
    symbol: str
    action: str
    delta_weight: float


class RebalanceResponse(BaseModel):
    generated_at: str
    instructions: list[RebalanceInstruction]


class PortfolioRefreshResponse(BaseModel):
    status: str
    refreshed_at: str
    positions_updated: int


class PortfolioHistoryPoint(BaseModel):
    recorded_at: str
    nav: float
    cash: float
    gross_exposure: float
    net_exposure: float
    benchmark: str
