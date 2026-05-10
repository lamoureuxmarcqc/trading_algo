from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user_roles: Mapped[list["UserRole"]] = relationship(back_populates="role", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[str] = mapped_column(String(512))
    mfa_secret: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    roles: Mapped[list["UserRole"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sessions: Mapped[list["UserSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class UserRole(Base):
    __tablename__ = "user_roles"

    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    )
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="roles")
    role: Mapped[Role] = relationship(back_populates="user_roles")


class UserSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"))
    refresh_token_hash: Mapped[str] = mapped_column(String(512))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="sessions")


class Symbol(Base):
    __tablename__ = "symbols"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    ticker: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    asset_class: Mapped[str] = mapped_column(String(64), default="equity")
    exchange: Mapped[str | None] = mapped_column(String(64), nullable=True)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    positions: Mapped[list["Position"]] = relationship(back_populates="symbol")
    orders: Mapped[list["Order"]] = relationship(back_populates="symbol")


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    broker_name: Mapped[str] = mapped_column(String(128))
    account_number: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    account_type: Mapped[str] = mapped_column(String(64))
    base_currency: Mapped[str] = mapped_column(String(8), default="USD")
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    cash_balances: Mapped[list["CashBalance"]] = relationship(
        back_populates="account", cascade="all, delete-orphan"
    )
    positions: Mapped[list["Position"]] = relationship(back_populates="account", cascade="all, delete-orphan")
    orders: Mapped[list["Order"]] = relationship(back_populates="account", cascade="all, delete-orphan")


class CashBalance(Base):
    __tablename__ = "cash_balances"

    account_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("accounts.id", ondelete="CASCADE"), primary_key=True
    )
    currency: Mapped[str] = mapped_column(String(8), primary_key=True)
    balance: Mapped[Decimal] = mapped_column(Numeric(20, 2), default=Decimal("0"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    account: Mapped[Account] = relationship(back_populates="cash_balances")


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    account_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("accounts.id", ondelete="CASCADE"))
    symbol_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("symbols.id", ondelete="CASCADE"))
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6))
    average_cost: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    market_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    market_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 2), nullable=True)
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(20, 2), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    account: Mapped[Account] = relationship(back_populates="positions")
    symbol: Mapped[Symbol] = relationship(back_populates="positions")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    account_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("accounts.id", ondelete="CASCADE"))
    symbol_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("symbols.id", ondelete="CASCADE"))
    side: Mapped[str] = mapped_column(String(16))
    order_type: Mapped[str] = mapped_column(String(32))
    tif: Mapped[str] = mapped_column(String(16), default="DAY")
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6))
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    broker_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    strategy_tag: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    account: Mapped[Account] = relationship(back_populates="orders")
    symbol: Mapped[Symbol] = relationship(back_populates="orders")
    fills: Mapped[list["Fill"]] = relationship(back_populates="order", cascade="all, delete-orphan")


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"
    __table_args__ = (
        UniqueConstraint("tenant_id", "operation", "idempotency_key", name="uq_idempotency_tenant_operation_key"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    operation: Mapped[str] = mapped_column(String(128), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(255))
    request_fingerprint: Mapped[str] = mapped_column(String(128))
    resource_type: Mapped[str] = mapped_column(String(64))
    resource_id: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class EventOutbox(Base):
    __tablename__ = "event_outbox"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    event_name: Mapped[str] = mapped_column(String(255), index=True)
    topic: Mapped[str] = mapped_column(String(255), index=True)
    event_version: Mapped[int] = mapped_column(default=1)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True, default="public")
    correlation_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    aggregate_type: Mapped[str] = mapped_column(String(128), index=True)
    aggregate_id: Mapped[str] = mapped_column(String(128), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    delivery_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text(), nullable=True)
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    order_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("orders.id", ondelete="CASCADE"))
    venue: Mapped[str] = mapped_column(String(64))
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6))
    price: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    fees: Mapped[Decimal] = mapped_column(Numeric(18, 6), default=Decimal("0"))
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    order: Mapped[Order] = relationship(back_populates="fills")


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    portfolio_id: Mapped[str] = mapped_column(String(128), index=True)
    nav: Mapped[Decimal] = mapped_column(Numeric(20, 2))
    cash: Mapped[Decimal] = mapped_column(Numeric(20, 2))
    gross_exposure: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    net_exposure: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    benchmark: Mapped[str] = mapped_column(String(32), default="SPY")
    base_currency: Mapped[str] = mapped_column(String(8), default="USD")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(64), index=True)
    entity_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    actor_email: Mapped[str] = mapped_column(String(255), default="system")
    details: Mapped[str | None] = mapped_column(Text(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
