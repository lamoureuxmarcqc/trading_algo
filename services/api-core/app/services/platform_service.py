from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, joinedload

from app.core.security import hash_password
from app.core.config import settings
from app.core.security import verify_password
from app.db.models import (
    Account,
    AuditLog,
    CashBalance,
    EventOutbox,
    Fill,
    IdempotencyKey,
    Order,
    PortfolioSnapshot,
    Position,
    Symbol,
    User,
    UserRole,
    UserSession,
)
from app.events.bus import event_bus
from app.events.contracts import DomainEvent
from app.schemas.admin import AdminUser, AuditLogEntry, DomainEventEntry, DomainEventSummary
from app.schemas.ai import ForecastResponse, RegimeResponse, SignalResponse
from app.schemas.auth import UserProfile
from app.schemas.portfolio import (
    PortfolioHistoryPoint,
    PortfolioOverview,
    PortfolioPerformance,
    PortfolioPosition,
    PortfolioRefreshResponse,
    RebalanceInstruction,
    RebalanceRequest,
    RebalanceResponse,
)
from app.schemas.research import FactorRank, ResearchIdea, SectorRotation
from app.schemas.risk import PortfolioRiskSnapshot, PositionRisk, ScenarioResult, ScenarioShock
from app.schemas.trading import Fill as FillSchema
from app.schemas.trading import OrderCreate, OrderResponse
from app.services.quant_service import quant_insights_service


@dataclass
class PositionView:
    symbol: str
    quantity: float
    average_cost: float
    market_price: float
    market_value: float
    daily_pnl: float
    unrealized_pnl: float
    currency: str


@dataclass
class OrderSubmissionResult:
    order: OrderResponse
    replayed: bool


class IdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused with a different request payload."""


class PlatformService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def authenticate_user(self, email: str, password: str) -> UserProfile | None:
        user = self.db.scalar(
            select(User)
            .where(User.email == email, User.is_active.is_(True))
            .options(joinedload(User.roles).joinedload(UserRole.role))
        )
        if not user or not verify_password(password, user.password_hash):
            return None
        return self._to_user_profile(user)

    def create_user_session(self, user_email: str, refresh_token: str) -> None:
        user = self.db.scalar(select(User).where(User.email == user_email, User.is_active.is_(True)))
        if not user:
            return
        session = UserSession(
            user_id=user.id,
            refresh_token_hash=hash_password(refresh_token),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        )
        self.db.add(session)
        self._audit(
            event_type="auth.login",
            entity_type="user",
            entity_id=user.id,
            actor_email=user.email,
            details="Interactive session created",
            commit=False,
        )
        self.db.commit()

    def get_current_user(self) -> UserProfile:
        user = self.db.scalar(
            select(User)
            .where(User.email == settings.default_user_email, User.is_active.is_(True))
            .options(joinedload(User.roles).joinedload(UserRole.role))
        )
        if user:
            return self._to_user_profile(user)

        fallback = self.db.scalar(
            select(User)
            .where(User.is_active.is_(True))
            .options(joinedload(User.roles).joinedload(UserRole.role))
        )
        if not fallback:
            raise ValueError("No active users found in database")
        return self._to_user_profile(fallback)

    def get_positions(self) -> list[PortfolioPosition]:
        return [self._position_schema(row) for row in self._position_rows(refresh_market_data=True)]

    def get_portfolio_overview(self, refresh_market_data: bool = True) -> PortfolioOverview:
        rows = self._position_rows(refresh_market_data=refresh_market_data)
        nav_from_positions = sum(row.market_value for row in rows)
        cash_total = self._cash_total()
        nav = nav_from_positions + cash_total
        gross_exposure = nav_from_positions / nav if nav else 0.0
        net_exposure = nav_from_positions / nav if nav else 0.0
        base_currency = self._base_currency()
        market_data_as_of = self._latest_position_update_iso()
        return PortfolioOverview(
            portfolio_id="family-office-master",
            nav=nav,
            cash=cash_total,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            base_currency=base_currency,
            benchmark=settings.default_benchmark,
            market_data_as_of=market_data_as_of,
            positions=[self._position_schema(row) for row in rows],
        )

    def get_portfolio_performance(self) -> PortfolioPerformance:
        rows = self._position_rows(refresh_market_data=True)
        total_value = sum(row.market_value for row in rows)
        total_cost = sum(row.average_cost * row.quantity for row in rows)
        unrealized = sum(row.unrealized_pnl for row in rows)
        year_return = (unrealized / total_cost) if total_cost else 0.0
        day_pnl = sum(row.daily_pnl for row in rows)
        day_return = (day_pnl / total_value) if total_value else 0.0
        month_return = day_return * 9
        return PortfolioPerformance(
            day_return=round(day_return, 4),
            month_return=round(month_return, 4),
            year_return=round(year_return, 4),
            alpha_vs_benchmark=round(year_return - 0.135, 4),
            sharpe_ratio=1.74 if total_value else 0.0,
            max_drawdown=-0.081 if total_value else 0.0,
        )

    def refresh_portfolio_market_data(
        self,
        force: bool = True,
        *,
        tenant_id: str = "public",
        correlation_id: str | None = None,
    ) -> PortfolioRefreshResponse:
        positions = self._refresh_positions_from_market(force=force)
        overview = self.get_portfolio_overview(refresh_market_data=False)
        refreshed_at = overview.market_data_as_of or datetime.now(timezone.utc).isoformat()
        self._upsert_portfolio_snapshot(overview)
        self._publish_domain_event(
            DomainEvent(
                event_name="com.terminal.portfolio.refreshed.v1",
                topic="terminal.portfolio.v1",
                aggregate_type="portfolio",
                aggregate_id=overview.portfolio_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
                payload={
                    "portfolio_id": overview.portfolio_id,
                    "positions_updated": len(positions),
                    "nav": round(overview.nav, 2),
                    "base_currency": overview.base_currency,
                    "benchmark": overview.benchmark,
                    "refreshed_at": refreshed_at,
                },
            )
        )
        self._audit(
            event_type="com.terminal.portfolio.refreshed.v1",
            entity_type="portfolio",
            entity_id=overview.portfolio_id,
            actor_email="system",
            details=f"Refreshed {len(positions)} positions from live market data",
        )
        return PortfolioRefreshResponse(
            status="refreshed",
            refreshed_at=refreshed_at,
            positions_updated=len(positions),
        )

    def get_portfolio_history(self, limit: int = 30) -> list[PortfolioHistoryPoint]:
        snapshots = (
            self.db.scalars(
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.portfolio_id == "family-office-master")
                .order_by(PortfolioSnapshot.created_at.desc())
                .limit(limit)
            )
            .all()
        )
        if not snapshots:
            overview = self.get_portfolio_overview(refresh_market_data=False)
            self._upsert_portfolio_snapshot(overview, force_insert=True)
            snapshots = (
                self.db.scalars(
                    select(PortfolioSnapshot)
                    .where(PortfolioSnapshot.portfolio_id == overview.portfolio_id)
                    .order_by(PortfolioSnapshot.created_at.desc())
                    .limit(limit)
                )
                .all()
            )

        return [
            PortfolioHistoryPoint(
                recorded_at=snapshot.created_at.isoformat(),
                nav=float(snapshot.nav),
                cash=float(snapshot.cash),
                gross_exposure=float(snapshot.gross_exposure),
                net_exposure=float(snapshot.net_exposure),
                benchmark=snapshot.benchmark,
            )
            for snapshot in reversed(snapshots)
        ]

    def rebalance_portfolio(self, payload: RebalanceRequest) -> RebalanceResponse:
        current_weights = self._current_weights()
        instructions: list[RebalanceInstruction] = []
        for target in payload.targets:
            current_weight = current_weights.get(target.symbol.upper(), 0.0)
            delta = round(target.target_weight - current_weight, 4)
            if abs(delta) < 0.01:
                continue
            instructions.append(
                RebalanceInstruction(
                    symbol=target.symbol.upper(),
                    action="BUY" if delta > 0 else "SELL",
                    delta_weight=delta,
                )
            )
        return RebalanceResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            instructions=instructions,
        )

    def create_order(
        self,
        payload: OrderCreate,
        *,
        idempotency_key: str | None = None,
        tenant_id: str = "public",
        correlation_id: str | None = None,
    ) -> OrderSubmissionResult:
        request_fingerprint = self._fingerprint_order_request(payload)
        if idempotency_key:
            replayed_order = self._resolve_idempotent_order(
                tenant_id=tenant_id,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
            )
            if replayed_order is not None:
                return OrderSubmissionResult(
                    order=self._order_schema(
                        replayed_order,
                        replayed_order.symbol.ticker,
                        replayed_order.account.broker_name,
                    ),
                    replayed=True,
                )

        account = self._default_account()
        symbol = self._get_or_create_symbol(payload.symbol.upper())
        order = Order(
            account_id=account.id,
            symbol_id=symbol.id,
            side=payload.side.upper(),
            order_type=payload.order_type.lower(),
            quantity=Decimal(str(payload.quantity)),
            limit_price=Decimal(str(payload.limit_price)) if payload.limit_price is not None else None,
            stop_price=Decimal(str(payload.stop_price)) if payload.stop_price is not None else None,
            status="accepted",
            broker_order_id=f"{payload.broker}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            strategy_tag=payload.strategy_tag,
        )
        self.db.add(order)
        self.db.flush()

        if idempotency_key:
            self.db.add(
                IdempotencyKey(
                    tenant_id=tenant_id,
                    operation="orders.create",
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
                    resource_type="order",
                    resource_id=order.id,
                )
            )

        event_name = "com.terminal.orders.created.v1"
        self._publish_domain_event(
            DomainEvent(
                event_name=event_name,
                topic="terminal.orders.v1",
                aggregate_type="order",
                aggregate_id=order.id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
                payload={
                    "order_id": order.id,
                    "symbol": symbol.ticker,
                    "side": order.side,
                    "order_type": order.order_type,
                    "quantity": float(order.quantity),
                    "limit_price": float(order.limit_price) if order.limit_price is not None else None,
                    "stop_price": float(order.stop_price) if order.stop_price is not None else None,
                    "status": order.status,
                    "broker": account.broker_name,
                    "strategy_tag": order.strategy_tag,
                    "idempotency_key": idempotency_key,
                    "created_at": order.created_at.isoformat() if order.created_at else None,
                },
            )
        )
        self._audit(
            event_type=event_name,
            entity_type="order",
            entity_id=order.id,
            actor_email=self.get_current_user().email,
            details=f"{order.side} {float(order.quantity):.0f} {symbol.ticker} via {account.broker_name}",
            commit=False,
        )
        self.db.commit()
        self.db.refresh(order)
        return OrderSubmissionResult(
            order=self._order_schema(order, symbol.ticker, account.broker_name),
            replayed=False,
        )

    def list_orders(self) -> list[OrderResponse]:
        orders = (
            self.db.execute(
                select(Order)
                .options(joinedload(Order.symbol), joinedload(Order.account), joinedload(Order.fills))
                .order_by(Order.created_at.desc())
            )
            .unique()
            .scalars()
            .all()
        )
        return [self._order_schema(order, order.symbol.ticker, order.account.broker_name) for order in orders]

    def cancel_order(
        self,
        order_id: str,
        *,
        tenant_id: str = "public",
        correlation_id: str | None = None,
    ) -> bool:
        order = self.db.get(Order, order_id)
        if not order:
            return False
        order.status = "cancelled"
        self.db.flush()
        self._publish_domain_event(
            DomainEvent(
                event_name="com.terminal.orders.cancelled.v1",
                topic="terminal.orders.v1",
                aggregate_type="order",
                aggregate_id=order.id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
                payload={
                    "order_id": order.id,
                    "status": order.status,
                    "symbol": order.symbol.ticker if order.symbol else None,
                    "cancelled_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
        self._audit(
            event_type="com.terminal.orders.cancelled.v1",
            entity_type="order",
            entity_id=order.id,
            actor_email=self.get_current_user().email,
            details="Order cancelled from institutional terminal",
            commit=False,
        )
        self.db.commit()
        return True

    def list_fills(self) -> list[FillSchema]:
        fills = self.db.scalars(
            select(Fill)
            .options(joinedload(Fill.order).joinedload(Order.symbol))
            .order_by(Fill.filled_at.desc())
        ).all()
        return [
            FillSchema(
                order_id=fill.order_id,
                symbol=fill.order.symbol.ticker,
                quantity=float(fill.quantity),
                price=float(fill.price),
                venue=fill.venue,
                filled_at=fill.filled_at.isoformat(),
            )
            for fill in fills
        ]

    def get_portfolio_risk(self) -> PortfolioRiskSnapshot:
        overview = self.get_portfolio_overview()
        metrics = quant_insights_service.compute_portfolio_risk(
            positions=overview.positions,
            gross_exposure=overview.gross_exposure,
            net_exposure=overview.net_exposure,
        )
        return PortfolioRiskSnapshot(**metrics)

    def get_position_risk(self) -> list[PositionRisk]:
        rows = self._position_rows(refresh_market_data=True)
        return [PositionRisk(**item) for item in quant_insights_service.compute_position_risks(rows)]

    def get_scenario_result(self, scenario_id: str) -> ScenarioResult | None:
        scenarios = {
            "2008": ScenarioResult(
                scenario_id="2008",
                name="Global Financial Crisis",
                estimated_pnl_impact=-184000,
                drawdown_impact=-0.137,
                shocks=[
                    ScenarioShock(factor="Equities", shock=-0.18, contribution=-124000),
                    ScenarioShock(factor="Credit Spread", shock=0.03, contribution=-40000),
                    ScenarioShock(factor="Volatility", shock=0.45, contribution=-20000),
                ],
            ),
            "covid": ScenarioResult(
                scenario_id="covid",
                name="COVID Liquidity Shock",
                estimated_pnl_impact=-121500,
                drawdown_impact=-0.092,
                shocks=[
                    ScenarioShock(factor="Equities", shock=-0.11, contribution=-89000),
                    ScenarioShock(factor="Rates", shock=-0.015, contribution=8500),
                    ScenarioShock(factor="USD", shock=0.06, contribution=-41000),
                ],
            ),
            "rates_up_2pct": ScenarioResult(
                scenario_id="rates_up_2pct",
                name="Rates +2%",
                estimated_pnl_impact=-64500,
                drawdown_impact=-0.048,
                shocks=[
                    ScenarioShock(factor="Rates", shock=0.02, contribution=-42000),
                    ScenarioShock(factor="Growth", shock=-0.06, contribution=-22500),
                ],
            ),
        }
        return scenarios.get(scenario_id)

    def get_signal(self, symbol: str) -> SignalResponse:
        position = next((row for row in self._position_rows(refresh_market_data=True) if row.symbol == symbol), None)
        payload = quant_insights_service.compute_signal(
            symbol=symbol,
            current_price=position.market_price if position else None,
        )
        return SignalResponse(**payload)

    def get_forecast(self, symbol: str) -> ForecastResponse:
        position = next((row for row in self._position_rows(refresh_market_data=True) if row.symbol == symbol), None)
        payload = quant_insights_service.compute_forecast(
            symbol=symbol,
            current_price=position.market_price if position else None,
        )
        return ForecastResponse(**payload)

    def get_regime(self) -> RegimeResponse:
        payload = quant_insights_service.compute_regime()
        return RegimeResponse(
            regime=str(payload["regime"]),
            confidence=float(payload["confidence"]),
            recommendation=str(payload["recommendation"]),
        )

    def get_research_screener(self) -> list[ResearchIdea]:
        return [ResearchIdea(**item) for item in quant_insights_service.build_research_screener()]

    def get_factor_ranking(self) -> list[FactorRank]:
        return [FactorRank(**item) for item in quant_insights_service.build_factor_ranking()]

    def get_sector_rotation(self) -> list[SectorRotation]:
        return [SectorRotation(**item) for item in quant_insights_service.build_sector_rotation()]

    def list_admin_users(self) -> list[AdminUser]:
        users = (
            self.db.execute(
                select(User)
                .options(joinedload(User.roles).joinedload(UserRole.role))
                .order_by(User.created_at.asc())
            )
            .unique()
            .scalars()
            .all()
        )
        return [
            AdminUser(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                role=user.roles[0].role.name if user.roles else "read-only",
                mfa_enabled=bool(user.mfa_secret),
                is_active=user.is_active,
                created_at=user.created_at.isoformat(),
            )
            for user in users
        ]

    def list_audit_logs(self, limit: int = 25) -> list[AuditLogEntry]:
        logs = self.db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)).all()
        return [
            AuditLogEntry(
                id=log.id,
                event_type=log.event_type,
                entity_type=log.entity_type,
                entity_id=log.entity_id,
                actor_email=log.actor_email,
                details=log.details,
                created_at=log.created_at.isoformat(),
            )
            for log in logs
        ]

    def list_domain_events(self, limit: int = 25) -> list[DomainEventEntry]:
        events = self.db.scalars(
            select(EventOutbox).order_by(EventOutbox.created_at.desc()).limit(limit)
        ).all()
        return [self._domain_event_entry(event) for event in events]

    def summarize_domain_events(self) -> DomainEventSummary:
        summary_rows = (
            self.db.execute(
                select(EventOutbox.delivery_status, func.count(EventOutbox.id))
                .group_by(EventOutbox.delivery_status)
            )
            .all()
        )
        summary = {"pending": 0, "failed": 0, "delivered": 0}
        for delivery_status, count in summary_rows:
            if delivery_status in summary:
                summary[delivery_status] = int(count)
        return DomainEventSummary(**summary)

    def _position_query(self) -> Select[tuple[Position]]:
        return select(Position).options(joinedload(Position.symbol), joinedload(Position.account))

    def _position_rows(self, refresh_market_data: bool = True) -> list[PositionView]:
        positions = self._position_entities(refresh_market_data=refresh_market_data)
        snapshot = quant_insights_service.get_live_market_snapshot([position.symbol.ticker for position in positions])
        rows = [
            PositionView(
                symbol=position.symbol.ticker,
                quantity=float(position.quantity),
                average_cost=float(position.average_cost),
                market_price=float(position.market_price or 0),
                market_value=float(position.market_value or 0),
                daily_pnl=self._daily_pnl_from_snapshot(position, snapshot),
                unrealized_pnl=float(position.unrealized_pnl or 0),
                currency=position.symbol.currency,
            )
            for position in positions
        ]
        return sorted(rows, key=lambda row: row.market_value, reverse=True)

    def _position_entities(self, refresh_market_data: bool = True) -> list[Position]:
        positions = self.db.scalars(self._position_query()).all()
        if refresh_market_data:
            refreshed = self._refresh_positions_from_market(force=False, positions=positions)
            if refreshed:
                positions = self.db.scalars(self._position_query()).all()
        return positions

    def _position_schema(self, row: PositionView) -> PortfolioPosition:
        return PortfolioPosition(
            symbol=row.symbol,
            quantity=row.quantity,
            average_cost=row.average_cost,
            market_price=row.market_price,
            market_value=row.market_value,
            daily_pnl=row.daily_pnl,
            unrealized_pnl=row.unrealized_pnl,
            currency=row.currency,
        )

    def _order_schema(self, order: Order, symbol: str, broker_name: str) -> OrderResponse:
        return OrderResponse(
            id=order.id,
            symbol=symbol,
            side=order.side,
            status=order.status,
            order_type=order.order_type,
            quantity=float(order.quantity),
            filled_quantity=float(sum(fill.quantity for fill in order.fills)) if order.fills else 0.0,
            limit_price=float(order.limit_price) if order.limit_price is not None else None,
            stop_price=float(order.stop_price) if order.stop_price is not None else None,
            broker=broker_name,
            created_at=order.created_at.isoformat(),
        )

    @staticmethod
    def _domain_event_entry(event: EventOutbox) -> DomainEventEntry:
        return DomainEventEntry(
            id=event.id,
            event_name=event.event_name,
            topic=event.topic,
            event_version=event.event_version,
            tenant_id=event.tenant_id,
            correlation_id=event.correlation_id,
            aggregate_type=event.aggregate_type,
            aggregate_id=event.aggregate_id,
            delivery_status=event.delivery_status,
            attempt_count=event.attempt_count,
            last_error=event.last_error,
            dispatched_at=event.dispatched_at.isoformat() if event.dispatched_at else None,
            payload=event.payload,
            created_at=event.created_at.isoformat(),
        )

    def _resolve_idempotent_order(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> Order | None:
        record = self.db.scalar(
            select(IdempotencyKey).where(
                IdempotencyKey.tenant_id == tenant_id,
                IdempotencyKey.operation == "orders.create",
                IdempotencyKey.idempotency_key == idempotency_key,
            )
        )
        if not record:
            return None
        if record.request_fingerprint != request_fingerprint:
            raise IdempotencyConflictError(
                "Idempotency-Key already used with a different order payload."
            )

        order = self.db.execute(
            select(Order)
            .options(joinedload(Order.symbol), joinedload(Order.account), joinedload(Order.fills))
            .where(Order.id == record.resource_id)
        ).unique().scalar_one_or_none()
        if order is None:
            raise IdempotencyConflictError(
                "Idempotency-Key points to a missing order resource. Manual investigation required."
            )
        return order

    @staticmethod
    def _fingerprint_order_request(payload: OrderCreate) -> str:
        canonical_payload = {
            "symbol": payload.symbol.upper(),
            "side": payload.side.upper(),
            "order_type": payload.order_type.lower(),
            "quantity": round(float(payload.quantity), 8),
            "limit_price": round(float(payload.limit_price), 8) if payload.limit_price is not None else None,
            "stop_price": round(float(payload.stop_price), 8) if payload.stop_price is not None else None,
            "broker": payload.broker,
            "strategy_tag": payload.strategy_tag,
        }
        encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _to_user_profile(self, user: User) -> UserProfile:
        role_name = "read-only"
        if user.roles:
            role_name = user.roles[0].role.name
        return UserProfile(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=role_name,
            mfa_enabled=bool(user.mfa_secret),
        )

    def _cash_total(self) -> float:
        total = self.db.scalar(select(func.coalesce(func.sum(CashBalance.balance), 0)))
        return float(total or 0)

    def _base_currency(self) -> str:
        account = self._default_account()
        return account.base_currency

    def _current_weights(self) -> dict[str, float]:
        rows = self._position_rows()
        total = sum(row.market_value for row in rows)
        if total <= 0:
            return {}
        return {row.symbol: row.market_value / total for row in rows}

    def _default_account(self) -> Account:
        account = self.db.scalar(select(Account).order_by(Account.created_at.asc()))
        if not account:
            raise ValueError("No trading account found in database")
        return account

    def _get_or_create_symbol(self, ticker: str) -> Symbol:
        symbol = self.db.scalar(select(Symbol).where(Symbol.ticker == ticker))
        if symbol:
            return symbol
        symbol = Symbol(ticker=ticker, asset_class="equity", exchange="SMART", currency="USD")
        self.db.add(symbol)
        self.db.flush()
        return symbol

    def _refresh_positions_from_market(
        self,
        force: bool = False,
        positions: list[Position] | None = None,
    ) -> list[Position]:
        positions = positions or self.db.scalars(self._position_query()).all()
        if not positions:
            return []
        if not force and not self._positions_stale(positions):
            return []

        symbols = [position.symbol.ticker for position in positions]
        snapshot = quant_insights_service.get_live_market_snapshot(symbols)
        updated_count = 0

        for position in positions:
            market = snapshot.get(position.symbol.ticker, {})
            latest_price = market.get("latest_price")
            if latest_price is None:
                continue

            quantity = Decimal(str(position.quantity))
            average_cost = Decimal(str(position.average_cost))
            latest_price_decimal = Decimal(str(round(float(latest_price), 6)))
            previous_close = market.get("previous_close")

            position.market_price = latest_price_decimal
            position.market_value = (latest_price_decimal * quantity).quantize(Decimal("0.01"))
            position.unrealized_pnl = ((latest_price_decimal - average_cost) * quantity).quantize(Decimal("0.01"))
            position.updated_at = datetime.now(timezone.utc)

            if previous_close is not None:
                previous_close_decimal = Decimal(str(round(float(previous_close), 6)))
                daily_pnl = ((latest_price_decimal - previous_close_decimal) * quantity).quantize(Decimal("0.01"))
                if daily_pnl == Decimal("-0.00"):
                    daily_pnl = Decimal("0.00")
            updated_count += 1

        if updated_count:
            self.db.commit()

        return positions

    @staticmethod
    def _positions_stale(positions: list[Position]) -> bool:
        if not positions:
            return False
        now = datetime.now(timezone.utc)
        threshold = settings.market_price_refresh_seconds
        for position in positions:
            if position.market_price is None or position.updated_at is None:
                return True
            age = (now - position.updated_at).total_seconds()
            if age >= threshold:
                return True
        return False

    def _latest_position_update_iso(self) -> str | None:
        latest = self.db.scalar(select(func.max(Position.updated_at)))
        return latest.isoformat() if latest else None

    def _upsert_portfolio_snapshot(
        self,
        overview: PortfolioOverview,
        force_insert: bool = False,
    ) -> None:
        latest = self.db.scalar(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.portfolio_id == overview.portfolio_id)
            .order_by(PortfolioSnapshot.created_at.desc())
            .limit(1)
        )
        if latest and not force_insert:
            age = (datetime.now(timezone.utc) - latest.created_at).total_seconds()
            if age < settings.market_price_refresh_seconds:
                latest.nav = Decimal(str(round(overview.nav, 2)))
                latest.cash = Decimal(str(round(overview.cash, 2)))
                latest.gross_exposure = Decimal(str(round(overview.gross_exposure, 6)))
                latest.net_exposure = Decimal(str(round(overview.net_exposure, 6)))
                latest.benchmark = overview.benchmark
                latest.base_currency = overview.base_currency
                self.db.commit()
                return

        snapshot = PortfolioSnapshot(
            portfolio_id=overview.portfolio_id,
            nav=Decimal(str(round(overview.nav, 2))),
            cash=Decimal(str(round(overview.cash, 2))),
            gross_exposure=Decimal(str(round(overview.gross_exposure, 6))),
            net_exposure=Decimal(str(round(overview.net_exposure, 6))),
            benchmark=overview.benchmark,
            base_currency=overview.base_currency,
        )
        self.db.add(snapshot)
        self.db.commit()

    def _audit(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str | None,
        actor_email: str,
        details: str,
        commit: bool = True,
    ) -> None:
        self.db.add(
            AuditLog(
                event_type=event_type,
                entity_type=entity_type,
                entity_id=entity_id,
                actor_email=actor_email,
                details=details,
            )
        )
        if commit:
            self.db.commit()

    def _publish_domain_event(self, event: DomainEvent) -> None:
        self.db.add(
            EventOutbox(
                event_name=event.event_name,
                topic=event.topic,
                event_version=event.event_version,
                tenant_id=event.tenant_id,
                correlation_id=event.correlation_id,
                aggregate_type=event.aggregate_type,
                aggregate_id=event.aggregate_id,
                payload=event.payload,
                delivery_status="pending",
            )
        )
        event_bus.publish(event)

    @staticmethod
    def _daily_pnl_from_snapshot(
        position: Position,
        snapshot: dict[str, dict[str, float | None]],
    ) -> float:
        market = snapshot.get(position.symbol.ticker, {})
        latest_price = market.get("latest_price")
        previous_close = market.get("previous_close")
        if latest_price is None or previous_close is None:
            return 0.0
        return round((float(latest_price) - float(previous_close)) * float(position.quantity), 2)
