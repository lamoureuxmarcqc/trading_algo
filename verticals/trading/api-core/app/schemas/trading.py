from pydantic import BaseModel, Field


class OrderCreate(BaseModel):
    symbol: str
    side: str
    order_type: str
    quantity: float = Field(gt=0)
    limit_price: float | None = None
    stop_price: float | None = None
    broker: str = "paper"
    strategy_tag: str | None = None


class OrderResponse(BaseModel):
    id: str
    symbol: str
    side: str
    status: str
    order_type: str
    quantity: float
    filled_quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    broker: str
    created_at: str


class Fill(BaseModel):
    order_id: str
    symbol: str
    quantity: float
    price: float
    venue: str
    filled_at: str

