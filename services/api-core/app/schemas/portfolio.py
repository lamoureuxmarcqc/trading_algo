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


class BarbellStrategyConfig(BaseModel):
    defensive_weight_target: float | None = Field(default=None, ge=0, le=0.95)
    opportunistic_weight_target: float | None = Field(default=None, ge=0, le=0.95)
    cash_buffer_target: float | None = Field(default=None, ge=0, le=0.50)
    max_positions_per_bucket: int = Field(default=3, ge=1, le=6)
    min_rebalance_delta: float = Field(default=0.01, ge=0, le=0.25)


class BarbellAllocationItem(BaseModel):
    symbol: str
    bucket: str
    role: str
    current_weight: float
    target_weight: float
    delta_weight: float
    buy_probability: float
    expected_return: float
    confidence_score: float
    rationale: str


class BarbellAllocationResponse(BaseModel):
    generated_at: str
    regime: str
    defensive_weight: float
    opportunistic_weight: float
    cash_buffer_weight: float
    rationale: str
    allocations: list[BarbellAllocationItem]
    rebalance_instructions: list[RebalanceInstruction]


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
