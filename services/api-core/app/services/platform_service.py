from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import logging
import math
import numpy as np
import pandas as pd

from sqlalchemy import Select, func, select
from sqlalchemy.exc import SQLAlchemyError
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
from app.schemas.admin import (
    AdminSymbolEntry,
    AdminSymbolMarketDataUpdate,
    AdminUser,
    AuditLogEntry,
    DomainEventEntry,
    DomainEventSummary,
)
from app.schemas.ai import ForecastResponse, RegimeResponse, SignalResponse
from app.schemas.auth import UserProfile
from app.schemas.portfolio import (
    AllocationOptimizationResponse,
    BarbellAllocationItem,
    BarbellAllocationResponse,
    BarbellStrategyConfig,
    PortfolioHistoryPoint,
    PortfolioOverview,
    PortfolioPerformance,
    PortfolioPosition,
    PortfolioRefreshResponse,
    RebalanceInstruction,
    RebalanceRequest,
    RebalanceResponse,
    OptimizationTarget,
)
from app.schemas.research import FactorRank, ResearchIdea, SectorRotation
from app.schemas.risk import (
    CorrelationMatrixResponse,
    PortfolioRiskSnapshot,
    PositionRisk,
    ScenarioImpactItem,
    ScenarioMacroMetric,
    ScenarioResult,
    ScenarioShock,
)
from app.schemas.trading import Fill as FillSchema
from app.schemas.trading import OrderCreate, OrderResponse
from app.schemas.terminal import (
    MonteCarloSimulationRequest,
    MonteCarloSimulationResponse,
    MonteCarloTrajectory,
    TerminalSnapshotResponse,
)
from app.services.quant_service import quant_insights_service
from trading_algo.strategy.barbell_strategy import (
    BarbellCandidate,
    BarbellConfig,
    BarbellStrategyEngine,
)


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
    market_data_enabled: bool
    market_data_symbol: str | None


@dataclass
class OrderSubmissionResult:
    order: OrderResponse
    replayed: bool


class IdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused with a different request payload."""


logger = logging.getLogger(__name__)
barbell_strategy_engine = BarbellStrategyEngine()

# In-memory override for barbell targets after "Apply Recommended Allocation".
_ACTIVE_BARBELL_TARGETS: dict[str, dict[str, float]] = {}
_ACTIVE_BARBELL_META: dict[str, dict[str, str]] = {}


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

    def get_barbell_allocation(
        self,
        config: BarbellStrategyConfig | None = None,
        *,
        tenant_id: str = "public",
    ) -> BarbellAllocationResponse:
        config = config or BarbellStrategyConfig()
        active_targets = _ACTIVE_BARBELL_TARGETS.get(tenant_id)
        if active_targets and not any(
            value is not None
            for value in (
                config.defensive_weight_target,
                config.opportunistic_weight_target,
                config.cash_buffer_target,
            )
        ):
            rows = self._position_rows(refresh_market_data=True)
            current_weights = self._current_weights()
            candidates = self._build_barbell_candidates(rows, current_weights)
            meta = _ACTIVE_BARBELL_META.get(tenant_id, {})
            barbell = self._barbell_response_from_targets(
                active_targets,
                candidates=candidates,
                regime=meta.get("regime", "optimized"),
                rationale=meta.get(
                    "rationale",
                    "Optimized barbell targets applied from the efficient frontier engine.",
                ),
            )
            return self._refresh_barbell_current_weights(barbell)

        rows = self._position_rows(refresh_market_data=True)
        overview = self.get_portfolio_overview(refresh_market_data=False)
        regime = self.get_regime()
        current_weights = self._current_weights()
        current_cash_weight = (overview.cash / overview.nav) if overview.nav > 0 else 1.0
        internal_config = BarbellConfig(
            defensive_weight_target=config.defensive_weight_target,
            opportunistic_weight_target=config.opportunistic_weight_target,
            cash_buffer_target=config.cash_buffer_target,
            max_positions_per_bucket=config.max_positions_per_bucket,
            min_rebalance_delta=config.min_rebalance_delta,
        )

        candidates = self._build_barbell_candidates(rows, current_weights)
        target_buckets = barbell_strategy_engine.resolve_bucket_targets(str(regime.regime), internal_config)
        defensive_selection = barbell_strategy_engine.select_candidates(
            candidates,
            "defensive",
            config.max_positions_per_bucket,
        )
        opportunistic_selection = barbell_strategy_engine.select_candidates(
            candidates,
            "opportunistic",
            config.max_positions_per_bucket,
        )

        target_weights = {
            **barbell_strategy_engine.allocate_bucket(defensive_selection, target_buckets["defensive"]),
            **barbell_strategy_engine.allocate_bucket(opportunistic_selection, target_buckets["opportunistic"]),
        }
        target_weights["CASH"] = target_buckets["cash"]

        candidate_map = {candidate.symbol: candidate for candidate in candidates}
        allocations: list[BarbellAllocationItem] = []
        bucket_order = {"defensive": 0, "opportunistic": 1, "cash": 2}

        for symbol, target_weight in target_weights.items():
            if symbol == "CASH":
                current_weight = current_cash_weight
                allocations.append(
                    BarbellAllocationItem(
                        symbol="CASH",
                        bucket="cash",
                        role="liquidity_reserve",
                        current_weight=round(current_weight, 4),
                        target_weight=round(target_weight, 4),
                        delta_weight=round(target_weight - current_weight, 4),
                        buy_probability=0.5,
                        expected_return=0.0,
                        confidence_score=1.0,
                        rationale="Cash reserve absorbs volatility and funds opportunistic redeployment.",
                    )
                )
                continue

            candidate = candidate_map[symbol]
            current_weight = current_weights.get(symbol, 0.0)
            allocations.append(
                BarbellAllocationItem(
                    symbol=symbol,
                    bucket=candidate.bucket,
                    role=candidate.role,
                    current_weight=round(current_weight, 4),
                    target_weight=round(target_weight, 4),
                    delta_weight=round(target_weight - current_weight, 4),
                    buy_probability=round(candidate.buy_probability, 4),
                    expected_return=round(candidate.expected_return, 4),
                    confidence_score=round(candidate.confidence_score, 4),
                    rationale=candidate.rationale,
                )
            )

        allocations.sort(
            key=lambda item: (bucket_order.get(item.bucket, 9), -item.target_weight, item.symbol)
        )

        rebalance_instructions = [
            RebalanceInstruction(
                symbol=item.symbol,
                action="BUY" if item.delta_weight > 0 else "SELL",
                delta_weight=item.delta_weight,
            )
            for item in allocations
            if item.symbol != "CASH" and abs(item.delta_weight) >= config.min_rebalance_delta
        ]

        return BarbellAllocationResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            regime=str(regime.regime),
            defensive_weight=round(target_buckets["defensive"], 4),
            opportunistic_weight=round(target_buckets["opportunistic"], 4),
            cash_buffer_weight=round(target_buckets["cash"], 4),
            rationale=self._barbell_rationale(
                regime=str(regime.regime),
                confidence=float(regime.confidence),
                target_buckets=target_buckets,
            ),
            allocations=allocations,
            rebalance_instructions=rebalance_instructions,
        )

    def rebalance_portfolio(
        self,
        payload: RebalanceRequest,
        *,
        tenant_id: str = "public",
        correlation_id: str | None = None,
    ) -> RebalanceResponse:
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
        orders: list[OrderResponse] = []
        notes: list[str] = []
        projected_cash = None
        projected_cash_weight = None
        projected_gross_exposure = None
        projected_net_exposure = None
        if payload.generate_orders:
            generated = self._generate_rebalance_orders(
                instructions=instructions,
                targets={target.symbol.upper(): target.target_weight for target in payload.targets},
                cash_buffer_target=payload.cash_buffer_target,
                min_trade_value=payload.min_trade_value,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            )
            orders = generated["orders"]
            notes = generated["notes"]
            projected_cash = generated["projected_cash"]
            projected_cash_weight = generated["projected_cash_weight"]
            projected_gross_exposure = generated["projected_gross_exposure"]
            projected_net_exposure = generated["projected_net_exposure"]
        return RebalanceResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            instructions=instructions,
            orders=orders,
            projected_cash=projected_cash,
            projected_cash_weight=projected_cash_weight,
            projected_gross_exposure=projected_gross_exposure,
            projected_net_exposure=projected_net_exposure,
            notes=notes,
        )

    def rebalance_barbell_portfolio(
        self,
        *,
        tenant_id: str = "public",
        correlation_id: str | None = None,
        cash_buffer_target: float = 0.15,
        min_trade_value: float = 500.0,
    ) -> RebalanceResponse:
        allocation = self.get_barbell_allocation(
            BarbellStrategyConfig(cash_buffer_target=cash_buffer_target, min_rebalance_delta=0.01),
            tenant_id=tenant_id,
        )
        targets = {
            item.symbol.upper(): item.target_weight
            for item in allocation.allocations
            if item.symbol != "CASH"
        }
        generated = self._generate_rebalance_orders(
            instructions=allocation.rebalance_instructions,
            targets=targets,
            cash_buffer_target=cash_buffer_target,
            min_trade_value=min_trade_value,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
        )
        return RebalanceResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            instructions=allocation.rebalance_instructions,
            orders=generated["orders"],
            projected_cash=generated["projected_cash"],
            projected_cash_weight=generated["projected_cash_weight"],
            projected_gross_exposure=generated["projected_gross_exposure"],
            projected_net_exposure=generated["projected_net_exposure"],
            notes=generated["notes"],
        )

    def run_monte_carlo_simulation(
        self,
        payload: MonteCarloSimulationRequest,
    ) -> MonteCarloSimulationResponse:
        from trading_algo.analytics.simulation import run_monte_carlo

        overview = self.get_portfolio_overview(refresh_market_data=False)
        rows = self._position_rows(refresh_market_data=False)
        returns = self._portfolio_return_frame(rows)
        symbols = list(returns.columns)
        if not symbols:
            symbols = [row.symbol for row in rows] or ["CASH"]
            returns = self._synthetic_return_frame(symbols)

        weights = self._weights_for_symbols(symbols, rows, payload.proposed_weights)
        np.random.seed(42)
        paths = run_monte_carlo(
            weights=weights,
            returns=returns,
            n_simulations=payload.n_paths,
            timeframe=payload.horizon_days,
            block_size=20,
            use_stochastic_vol=True,
            vol_kappa=0.15,
            vol_theta=1.0,
            vol_sigma=0.25,
        )
        nav_paths = (paths / 100.0) * max(float(overview.nav), 1.0)
        trajectory = [
            MonteCarloTrajectory(
                day=index + 1,
                p5_nav=round(float(np.percentile(nav_paths[index, :], 5)), 2),
                p50_nav=round(float(np.percentile(nav_paths[index, :], 50)), 2),
                p95_nav=round(float(np.percentile(nav_paths[index, :], 95)), 2),
            )
            for index in range(nav_paths.shape[0])
        ]

        final_returns = (nav_paths[-1, :] / max(float(overview.nav), 1.0)) - 1.0
        var_95 = float(np.percentile(final_returns, 5))
        tail = final_returns[final_returns <= var_95]
        daily_returns = (nav_paths[1:, :] / np.maximum(nav_paths[:-1, :], 1e-9)) - 1.0
        mean_daily = float(np.mean(daily_returns))
        std_daily = float(np.std(daily_returns))
        expected_return = float(np.mean(final_returns))
        sharpe = ((mean_daily * 252) / (std_daily * math.sqrt(252))) if std_daily > 0 else 0.0

        return MonteCarloSimulationResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            n_paths=payload.n_paths,
            horizon_days=payload.horizon_days,
            nav=round(float(overview.nav), 2),
            expected_annual_return=round(expected_return, 4),
            simulated_sharpe_ratio=round(float(sharpe), 4),
            var_95=round(var_95, 4),
            cvar_95=round(float(tail.mean()) if len(tail) else var_95, 4),
            trajectory=trajectory,
            symbols=symbols,
            methodology="trading_algo.analytics.simulation.run_monte_carlo using 20-day volatility, beta-aware portfolio weights and historical/fallback correlations.",
        )

    def optimize_barbell_allocation(
        self,
        apply_to_barbell: bool = False,
        *,
        tenant_id: str = "public",
    ) -> AllocationOptimizationResponse:
        overview = self.get_portfolio_overview(refresh_market_data=False)
        rows = self._position_rows(refresh_market_data=False)
        current_weights = self._current_weights()
        candidates = self._build_barbell_candidates(rows, current_weights)
        selected = sorted(
            candidates,
            key=lambda item: (
                item.confidence_score * 0.35
                + item.buy_probability * 0.35
                + max(item.expected_return, 0.0) * 2.0
                + item.liquidity_score * 0.10
            ),
            reverse=True,
        )[:8]
        symbols = [candidate.symbol for candidate in selected]
        returns = self._portfolio_return_frame(rows, symbols=symbols)
        if returns.empty:
            returns = self._synthetic_return_frame(symbols)
        returns = returns.reindex(columns=symbols).fillna(0.0)

        annual_returns = np.array([max(candidate.expected_return, 0.01) for candidate in selected])
        annual_cov = returns.cov().fillna(0.0).to_numpy() * 252
        if not np.any(annual_cov):
            annual_vols = np.array([max(0.08, 1.0 - candidate.volatility_score) for candidate in selected])
            corr = np.full((len(symbols), len(symbols)), 0.35)
            np.fill_diagonal(corr, 1.0)
            annual_cov = np.outer(annual_vols, annual_vols) * corr

        cash_weight = 0.15
        risky_weight = 1.0 - cash_weight
        rng = np.random.default_rng(123)
        best: dict[str, object] | None = None
        best_relaxed: dict[str, object] | None = None
        for _ in range(5000):
            risky = rng.dirichlet(np.ones(len(symbols))) * risky_weight
            ret = float(risky @ annual_returns + cash_weight * 0.02)
            vol = float(math.sqrt(max(risky @ annual_cov @ risky, 1e-12)))
            sharpe = (ret - 0.02) / vol if vol > 0 else 0.0
            var_95 = ret - 1.65 * vol
            cvar_95 = ret - 2.06 * vol
            item = {
                "weights": risky,
                "ret": ret,
                "vol": vol,
                "sharpe": sharpe,
                "var_95": var_95,
                "cvar_95": cvar_95,
            }
            if best_relaxed is None or float(item["sharpe"]) > float(best_relaxed["sharpe"]):
                best_relaxed = item
            if var_95 >= -0.015 and (best is None or sharpe > float(best["sharpe"])):
                best = item

        notes: list[str] = []
        status = "ok"
        if best is None:
            best = best_relaxed
            status = "relaxed_var_constraint"
            notes.append("No sampled portfolio satisfied VaR >= -1.5%; returning the best Sharpe candidate.")

        assert best is not None
        weights = np.array(best["weights"], dtype=float)
        optimized_targets = {
            symbol: round(float(weight), 4)
            for symbol, weight in zip(symbols, weights, strict=True)
            if weight >= 0.0025
        }
        optimized_targets["CASH"] = cash_weight
        barbell = self._barbell_response_from_targets(
            optimized_targets,
            candidates=selected,
            regime="optimized",
            rationale=(
                "Markowitz max-Sharpe posture with a 15% cash reserve and VaR guardrail. "
                "Use Apply Recommended Allocation to make these the visible terminal targets."
            ),
        )
        targets = [
            OptimizationTarget(
                symbol=candidate.symbol,
                current_weight=round(float(current_weights.get(candidate.symbol, 0.0)), 4),
                recommended_weight=round(float(optimized_targets.get(candidate.symbol, 0.0)), 4),
                delta_weight=round(float(optimized_targets.get(candidate.symbol, 0.0)) - float(current_weights.get(candidate.symbol, 0.0)), 4),
                expected_return=round(float(candidate.expected_return), 4),
                volatility=round(float(max(0.08, 1.0 - candidate.volatility_score)), 4),
                bucket=candidate.bucket,
            )
            for candidate in selected
            if candidate.symbol in optimized_targets
        ]
        if apply_to_barbell:
            _ACTIVE_BARBELL_TARGETS[tenant_id] = optimized_targets
            _ACTIVE_BARBELL_META[tenant_id] = {
                "regime": "optimized",
                "rationale": barbell.rationale,
            }
            notes.append("Optimized targets persisted as the active barbell posture for this terminal session.")

        return AllocationOptimizationResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=status,
            objective="maximize_sharpe_with_var_guardrail",
            expected_annual_return=round(float(best["ret"]), 4),
            expected_annual_volatility=round(float(best["vol"]), 4),
            simulated_sharpe_ratio=round(float(best["sharpe"]), 4),
            var_95=round(float(best["var_95"]), 4),
            cvar_95=round(float(best["cvar_95"]), 4),
            cash_buffer_weight=cash_weight,
            targets=targets,
            barbell=barbell,
            notes=notes,
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
        rows = self._position_rows(refresh_market_data=False)
        metrics = quant_insights_service.compute_portfolio_risk(
            positions=rows,
            gross_exposure=overview.gross_exposure,
            net_exposure=overview.net_exposure,
        )
        return PortfolioRiskSnapshot(**metrics)

    def get_position_risk(self) -> list[PositionRisk]:
        rows = self._position_rows(refresh_market_data=True)
        return [PositionRisk(**item) for item in quant_insights_service.compute_position_risks(rows)]

    def get_scenario_result(self, scenario_id: str) -> ScenarioResult | None:
        scenarios = {scenario.scenario_id: scenario for scenario in self.list_scenarios()}
        return scenarios.get(scenario_id)

    def list_scenarios(self) -> list[ScenarioResult]:
        overview = self.get_portfolio_overview()
        rows = self._position_rows(refresh_market_data=True)
        scenario_results = [self._build_scenario_result(config, overview, rows) for config in self._scenario_catalog()]
        return sorted(scenario_results, key=lambda item: item.drawdown_impact)

    def get_portfolio_correlation_matrix(self) -> CorrelationMatrixResponse:
        rows = self._position_rows(refresh_market_data=True)
        payload = quant_insights_service.compute_correlation_matrix(rows)
        return CorrelationMatrixResponse(**payload)

    def get_signal(self, symbol: str) -> SignalResponse:
        position = next((row for row in self._position_rows(refresh_market_data=True) if row.symbol == symbol), None)
        payload = quant_insights_service.compute_signal(
            symbol=symbol,
            current_price=position.market_price if position else None,
            market_data_symbol=position.market_data_symbol if position else None,
            market_data_enabled=position.market_data_enabled if position else True,
        )
        return SignalResponse(**payload)

    def get_forecast(self, symbol: str) -> ForecastResponse:
        position = next((row for row in self._position_rows(refresh_market_data=True) if row.symbol == symbol), None)
        payload = quant_insights_service.compute_forecast(
            symbol=symbol,
            current_price=position.market_price if position else None,
            market_data_symbol=position.market_data_symbol if position else None,
            market_data_enabled=position.market_data_enabled if position else True,
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

    def get_terminal_snapshot(self) -> TerminalSnapshotResponse:
        overview = self.get_portfolio_overview(refresh_market_data=False)
        performance = self.get_portfolio_performance()
        risk = self.get_portfolio_risk()
        regime = self.get_regime()
        history = self.get_portfolio_history(limit=30)
        rows = self._position_rows(refresh_market_data=False)
        signals, forecasts = self._terminal_opportunity_tape(rows, regime.regime)
        research, factors, sectors = self._terminal_research_bundle(rows)

        return TerminalSnapshotResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            portfolio=overview,
            performance=performance,
            risk=risk,
            regime=regime,
            history=history,
            signals=signals,
            forecasts=forecasts,
            scenarios=self.list_scenarios(),
            barbell=self.get_barbell_allocation(),
            correlation_matrix=self.get_portfolio_correlation_matrix(),
            position_risk=self.get_position_risk(),
            orders=self.list_orders(),
            fills=self.list_fills(),
            research=research,
            factors=factors,
            sectors=sectors,
            users=self.list_admin_users(),
            audit_logs=self.list_audit_logs(limit=25),
            event_summary=self.summarize_domain_events(),
        )

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

    def list_admin_symbols(self, unresolved_only: bool = False, limit: int = 100) -> list[AdminSymbolEntry]:
        rows = (
            self.db.execute(
                select(
                    Symbol,
                    func.count(Position.id).label("position_count"),
                    func.coalesce(func.sum(Position.market_value), 0).label("total_market_value"),
                    func.max(Position.market_price).label("last_price"),
                )
                .outerjoin(Position, Position.symbol_id == Symbol.id)
                .group_by(Symbol.id)
                .order_by(
                    func.coalesce(func.sum(Position.market_value), 0).desc(),
                    Symbol.ticker.asc(),
                )
                .limit(limit)
            )
            .all()
        )

        entries = [
            AdminSymbolEntry(
                id=symbol.id,
                ticker=symbol.ticker,
                asset_class=symbol.asset_class,
                exchange=symbol.exchange,
                currency=symbol.currency,
                market_data_ticker=self._effective_market_data_ticker(symbol),
                market_data_enabled=bool(symbol.market_data_enabled),
                position_count=int(position_count or 0),
                total_market_value=float(total_market_value or 0.0),
                last_price=float(last_price) if last_price is not None else None,
            )
            for symbol, position_count, total_market_value, last_price in rows
        ]
        if unresolved_only:
            return [
                entry
                for entry in entries
                if not entry.market_data_enabled
            ]
        return entries

    def update_symbol_market_data(
        self,
        symbol_id: str,
        payload: AdminSymbolMarketDataUpdate,
    ) -> AdminSymbolEntry:
        symbol = self.db.get(Symbol, symbol_id)
        if not symbol:
            raise ValueError("Symbol not found")

        symbol.market_data_enabled = payload.market_data_enabled
        normalized_market_ticker = payload.market_data_ticker.strip().upper() if payload.market_data_ticker else None
        symbol.market_data_ticker = normalized_market_ticker if payload.market_data_enabled else None
        if payload.market_data_enabled and not symbol.market_data_ticker:
            symbol.market_data_ticker = symbol.ticker

        self.db.commit()
        self.db.refresh(symbol)

        position_count = self.db.scalar(select(func.count(Position.id)).where(Position.symbol_id == symbol.id)) or 0
        total_market_value = (
            self.db.scalar(select(func.coalesce(func.sum(Position.market_value), 0)).where(Position.symbol_id == symbol.id))
            or 0
        )
        last_price = self.db.scalar(select(func.max(Position.market_price)).where(Position.symbol_id == symbol.id))
        return AdminSymbolEntry(
            id=symbol.id,
            ticker=symbol.ticker,
            asset_class=symbol.asset_class,
            exchange=symbol.exchange,
            currency=symbol.currency,
            market_data_ticker=self._effective_market_data_ticker(symbol),
            market_data_enabled=bool(symbol.market_data_enabled),
            position_count=int(position_count),
            total_market_value=float(total_market_value or 0.0),
            last_price=float(last_price) if last_price is not None else None,
        )

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

    def _generate_rebalance_orders(
        self,
        *,
        instructions: list[RebalanceInstruction],
        targets: dict[str, float],
        cash_buffer_target: float,
        min_trade_value: float,
        tenant_id: str,
        correlation_id: str | None,
    ) -> dict[str, object]:
        overview = self.get_portfolio_overview(refresh_market_data=False)
        rows = self._position_rows(refresh_market_data=False)
        row_map = {row.symbol: row for row in rows}
        price_map = {row.symbol: max(float(row.market_price), 0.01) for row in rows}
        generated_orders: list[OrderResponse] = []
        notes: list[str] = []

        required_cash = max(float(overview.nav) * cash_buffer_target, 0.0)
        sells = [item for item in instructions if item.action.upper() == "SELL"]
        buys = [item for item in instructions if item.action.upper() == "BUY"]
        sell_notional = 0.0

        for instruction in sorted(sells, key=lambda item: abs(item.delta_weight), reverse=True):
            row = row_map.get(instruction.symbol)
            if not row:
                notes.append(f"Skipped SELL {instruction.symbol}: no current position.")
                continue
            price = price_map.get(instruction.symbol, max(row.market_price, 0.01))
            notional = min(abs(float(instruction.delta_weight)) * float(overview.nav), row.quantity * price)
            if notional < min_trade_value:
                notes.append(f"Skipped SELL {instruction.symbol}: trade value below threshold.")
                continue
            quantity = max(round(notional / price, 4), 0.0001)
            order = self.create_order(
                OrderCreate(
                    symbol=instruction.symbol,
                    side="SELL",
                    order_type="limit",
                    quantity=quantity,
                    limit_price=round(price * 0.995, 2),
                    broker="paper",
                    strategy_tag="barbell_rebalance",
                ),
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            ).order
            generated_orders.append(order)
            sell_notional += notional

        buy_budget = max(0.0, float(overview.cash) + sell_notional - required_cash)
        planned_buys = [
            (instruction, max(float(instruction.delta_weight), 0.0) * float(overview.nav))
            for instruction in buys
        ]
        total_requested_buy = sum(value for _, value in planned_buys)
        if total_requested_buy > buy_budget and total_requested_buy > 0:
            notes.append(
                f"BUY orders scaled to preserve {cash_buffer_target:.0%} cash buffer "
                f"(${buy_budget:,.0f} budget vs ${total_requested_buy:,.0f} requested)."
            )
        scale = min(1.0, buy_budget / total_requested_buy) if total_requested_buy > 0 else 0.0
        buy_notional = 0.0

        for instruction, requested_notional in sorted(planned_buys, key=lambda item: item[1], reverse=True):
            notional = requested_notional * scale
            if notional < min_trade_value:
                notes.append(f"Skipped BUY {instruction.symbol}: trade value below threshold or cash buffer constraint.")
                continue
            price = self._reference_price_for_symbol(instruction.symbol, price_map)
            quantity = max(round(notional / price, 4), 0.0001)
            order = self.create_order(
                OrderCreate(
                    symbol=instruction.symbol,
                    side="BUY",
                    order_type="limit",
                    quantity=quantity,
                    limit_price=round(price * 1.005, 2),
                    broker="paper",
                    strategy_tag="barbell_rebalance",
                ),
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            ).order
            generated_orders.append(order)
            buy_notional += notional

        projected_cash = float(overview.cash) + sell_notional - buy_notional
        projected_exposure = max(float(overview.nav) - projected_cash, 0.0)
        if not generated_orders:
            notes.append("No rebalance orders generated; targets are within threshold or cash buffer blocks buys.")
        return {
            "orders": generated_orders,
            "notes": notes,
            "projected_cash": round(projected_cash, 2),
            "projected_cash_weight": round(projected_cash / float(overview.nav), 4) if overview.nav else None,
            "projected_gross_exposure": round(projected_exposure / float(overview.nav), 4) if overview.nav else None,
            "projected_net_exposure": round(projected_exposure / float(overview.nav), 4) if overview.nav else None,
            "targets": targets,
        }

    def _portfolio_return_frame(
        self,
        rows: list[PositionView],
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        row_map = {row.symbol: row for row in rows}
        requested = symbols or [row.symbol for row in rows]
        returns_map: dict[str, pd.Series] = {}
        for symbol in requested:
            row = row_map.get(symbol)
            try:
                returns = quant_insights_service._returns_series(
                    symbol,
                    period="1y",
                    market_data_symbol=row.market_data_symbol if row else symbol,
                    market_data_enabled=row.market_data_enabled if row else True,
                )
            except Exception:
                returns = pd.Series(dtype=float)
            if not returns.empty:
                returns_map[symbol] = returns.tail(252)
        if not returns_map:
            return pd.DataFrame()
        frame = pd.DataFrame(returns_map).dropna(how="all").ffill().dropna()
        return frame.tail(252)

    def _synthetic_return_frame(self, symbols: list[str], periods: int = 252) -> pd.DataFrame:
        candidates = {item["symbol"]: item for item in self._barbell_universe_catalog()}
        rng = np.random.default_rng(99)
        vols = np.array([
            max(0.08, 1.0 - float(candidates.get(symbol, {}).get("volatility_score", 0.70)))
            for symbol in symbols
        ])
        annual_returns = np.array([
            float(candidates.get(symbol, {}).get("expected_return", 0.05))
            for symbol in symbols
        ])
        corr = np.full((len(symbols), len(symbols)), 0.35)
        np.fill_diagonal(corr, 1.0)
        daily_cov = np.outer(vols / math.sqrt(252), vols / math.sqrt(252)) * corr
        daily_mean = annual_returns / 252
        data = rng.multivariate_normal(daily_mean, daily_cov, size=periods)
        return pd.DataFrame(data, columns=symbols)

    @staticmethod
    def _weights_for_symbols(
        symbols: list[str],
        rows: list[PositionView],
        proposed_weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        if proposed_weights:
            weights = np.array([max(float(proposed_weights.get(symbol, 0.0)), 0.0) for symbol in symbols])
        else:
            total = sum(max(row.market_value, 0.0) for row in rows if row.symbol in symbols)
            weights = np.array([
                (max(next((row.market_value for row in rows if row.symbol == symbol), 0.0), 0.0) / total)
                if total > 0
                else 1.0 / len(symbols)
                for symbol in symbols
            ])
        total_weight = float(weights.sum())
        if total_weight <= 0:
            return np.array([1.0 / len(symbols)] * len(symbols))
        return weights / total_weight

    def _refresh_barbell_current_weights(self, barbell: BarbellAllocationResponse) -> BarbellAllocationResponse:
        overview = self.get_portfolio_overview(refresh_market_data=False)
        current_weights = self._current_weights()
        current_cash_weight = (overview.cash / overview.nav) if overview.nav > 0 else 1.0
        allocations: list[BarbellAllocationItem] = []
        for item in barbell.allocations:
            current_weight = current_cash_weight if item.symbol == "CASH" else current_weights.get(item.symbol, 0.0)
            allocations.append(
                item.model_copy(
                    update={
                        "current_weight": round(current_weight, 4),
                        "delta_weight": round(item.target_weight - current_weight, 4),
                    }
                )
            )
        rebalance_instructions = [
            RebalanceInstruction(
                symbol=item.symbol,
                action="BUY" if item.delta_weight > 0 else "SELL",
                delta_weight=item.delta_weight,
            )
            for item in allocations
            if item.symbol != "CASH" and abs(item.delta_weight) >= 0.01
        ]
        return barbell.model_copy(
            update={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "allocations": allocations,
                "rebalance_instructions": rebalance_instructions,
            }
        )

    def _barbell_response_from_targets(
        self,
        targets: dict[str, float],
        *,
        candidates: list[BarbellCandidate],
        regime: str,
        rationale: str,
    ) -> BarbellAllocationResponse:
        overview = self.get_portfolio_overview(refresh_market_data=False)
        current_weights = self._current_weights()
        candidate_map = {candidate.symbol: candidate for candidate in candidates}
        allocations: list[BarbellAllocationItem] = []
        for symbol, target_weight in targets.items():
            if symbol == "CASH":
                current_weight = (overview.cash / overview.nav) if overview.nav > 0 else 1.0
                allocations.append(
                    BarbellAllocationItem(
                        symbol="CASH",
                        bucket="cash",
                        role="liquidity_reserve",
                        current_weight=round(current_weight, 4),
                        target_weight=round(target_weight, 4),
                        delta_weight=round(target_weight - current_weight, 4),
                        buy_probability=0.5,
                        expected_return=0.0,
                        confidence_score=1.0,
                        rationale="Cash reserve preserves the strategic liquidity guardrail.",
                    )
                )
                continue
            candidate = candidate_map[symbol]
            current_weight = current_weights.get(symbol, 0.0)
            allocations.append(
                BarbellAllocationItem(
                    symbol=symbol,
                    bucket=candidate.bucket,
                    role=candidate.role,
                    current_weight=round(current_weight, 4),
                    target_weight=round(target_weight, 4),
                    delta_weight=round(target_weight - current_weight, 4),
                    buy_probability=round(candidate.buy_probability, 4),
                    expected_return=round(candidate.expected_return, 4),
                    confidence_score=round(candidate.confidence_score, 4),
                    rationale=candidate.rationale,
                )
            )
        rebalance_instructions = [
            RebalanceInstruction(
                symbol=item.symbol,
                action="BUY" if item.delta_weight > 0 else "SELL",
                delta_weight=item.delta_weight,
            )
            for item in allocations
            if item.symbol != "CASH" and abs(item.delta_weight) >= 0.01
        ]
        return BarbellAllocationResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            regime=regime,
            defensive_weight=round(
                sum(item.target_weight for item in allocations if item.bucket == "defensive"), 4
            ),
            opportunistic_weight=round(
                sum(item.target_weight for item in allocations if item.bucket == "opportunistic"), 4
            ),
            cash_buffer_weight=round(float(targets.get("CASH", 0.0)), 4),
            rationale=rationale,
            allocations=sorted(allocations, key=lambda item: (item.bucket, -item.target_weight, item.symbol)),
            rebalance_instructions=rebalance_instructions,
        )

    def _reference_price_for_symbol(self, symbol: str, price_map: dict[str, float]) -> float:
        if symbol in price_map and price_map[symbol] > 0:
            return price_map[symbol]
        symbol_row = self.db.scalar(select(Symbol).where(Symbol.ticker == symbol))
        if symbol_row:
            last_price = self.db.scalar(select(func.max(Position.market_price)).where(Position.symbol_id == symbol_row.id))
            if last_price:
                return max(float(last_price), 0.01)
        catalog = {item["symbol"]: item for item in self._barbell_universe_catalog()}
        expected_return = float(catalog.get(symbol, {}).get("expected_return", 0.05))
        return round(100.0 * (1 + expected_return), 2)

    def _position_query(self) -> Select[tuple[Position]]:
        return select(Position).options(joinedload(Position.symbol), joinedload(Position.account))

    def _position_rows(self, refresh_market_data: bool = True) -> list[PositionView]:
        positions = self._position_entities(refresh_market_data=refresh_market_data)
        snapshot = (
            quant_insights_service.get_live_market_snapshot_for_requests(
                {
                    position.symbol.ticker: self._market_data_symbol(position.symbol)
                    for position in positions
                }
            )
            if refresh_market_data
            else {}
        )
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
                market_data_enabled=bool(position.symbol.market_data_enabled),
                market_data_symbol=self._market_data_symbol(position.symbol),
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

    def _build_barbell_candidates(
        self,
        rows: list[PositionView],
        current_weights: dict[str, float],
    ) -> list[BarbellCandidate]:
        universe = {item["symbol"]: item for item in self._barbell_universe_catalog()}
        live_symbols = {row.symbol for row in rows}

        for row in rows:
            universe.setdefault(
                row.symbol,
                {
                    "symbol": row.symbol,
                    "bucket": self._infer_barbell_bucket(row.symbol),
                    "role": "existing_holding",
                    "rationale": f"Existing portfolio exposure kept in the barbell universe for {row.symbol}.",
                    "buy_probability": 0.58,
                    "expected_return": 0.05,
                    "confidence_score": 0.62,
                    "quality_score": 0.60,
                    "volatility_score": 0.55,
                    "liquidity_score": 0.70,
                },
            )

        candidates: list[BarbellCandidate] = []
        for symbol, descriptor in universe.items():
            buy_probability = float(descriptor["buy_probability"])
            expected_return = float(descriptor["expected_return"])
            confidence_score = float(descriptor["confidence_score"])
            quality_score = float(descriptor["quality_score"])
            volatility_score = float(descriptor["volatility_score"])
            liquidity_score = float(descriptor["liquidity_score"])

            if symbol in live_symbols:
                try:
                    row = next((item for item in rows if item.symbol == symbol), None)
                    if row and row.market_data_enabled and row.market_data_symbol:
                        signal = quant_insights_service.compute_signal(
                            symbol=symbol,
                            current_price=row.market_price,
                            market_data_symbol=row.market_data_symbol,
                            market_data_enabled=row.market_data_enabled,
                        )
                        forecast = quant_insights_service.compute_forecast(
                            symbol=symbol,
                            current_price=row.market_price,
                            market_data_symbol=row.market_data_symbol,
                            market_data_enabled=row.market_data_enabled,
                        )
                        factor_metrics = quant_insights_service._factor_metrics(
                            symbol,
                            market_data_symbol=row.market_data_symbol,
                            market_data_enabled=row.market_data_enabled,
                        )
                        liquidity_score = float(
                            quant_insights_service._liquidity_score(
                                symbol,
                                market_data_symbol=row.market_data_symbol,
                                market_data_enabled=row.market_data_enabled,
                            )
                        )
                        buy_probability = float(signal["buy_probability"])
                        expected_return = float(forecast["expected_return"])
                        confidence_score = float(signal["confidence_score"])
                        quality_score = float(factor_metrics["quality_score"])
                        volatility_score = float(factor_metrics["volatility_score"])
                except Exception:
                    logger.debug("Falling back to static barbell profile for %s", symbol, exc_info=True)

            candidates.append(
                BarbellCandidate(
                    symbol=symbol,
                    bucket=str(descriptor["bucket"]),
                    role=str(descriptor["role"]),
                    current_weight=float(current_weights.get(symbol, 0.0)),
                    buy_probability=buy_probability,
                    expected_return=expected_return,
                    confidence_score=confidence_score,
                    quality_score=quality_score,
                    volatility_score=volatility_score,
                    liquidity_score=liquidity_score,
                    rationale=str(descriptor["rationale"]),
                )
            )
        return candidates

    def _terminal_research_bundle(
        self,
        rows: list[PositionView],
    ) -> tuple[list[ResearchIdea], list[FactorRank], list[SectorRotation]]:
        current_weights = self._current_weights()
        candidates = self._build_barbell_candidates(rows, current_weights)
        sector_map = self._symbol_sector_map()
        price_map = {row.symbol: row.market_price for row in rows}

        screener: list[ResearchIdea] = []
        factors: list[FactorRank] = []
        grouped: dict[str, list[ResearchIdea]] = {}

        for candidate in candidates:
            sector = sector_map.get(candidate.symbol, "diversified").replace("_", " ").title()
            momentum_score = max(
                0.05,
                min(candidate.buy_probability * 0.65 + max(candidate.expected_return, 0.0) * 2.0, 0.95),
            )
            overall_score = (
                momentum_score * 0.45
                + candidate.quality_score * 0.35
                + candidate.volatility_score * 0.20
            )
            idea = ResearchIdea(
                symbol=candidate.symbol,
                sector=sector,
                price=round(float(price_map.get(candidate.symbol, 100.0)), 2),
                buy_probability=round(candidate.buy_probability, 4),
                expected_return=round(candidate.expected_return, 4),
                confidence_score=round(candidate.confidence_score, 4),
                factor_score=round(overall_score, 4),
                market_regime="risk_on" if candidate.bucket == "opportunistic" else "defensive",
            )
            screener.append(idea)
            grouped.setdefault(sector, []).append(idea)
            factors.append(
                FactorRank(
                    symbol=candidate.symbol,
                    sector=sector,
                    momentum_score=round(momentum_score, 4),
                    quality_score=round(candidate.quality_score, 4),
                    volatility_score=round(candidate.volatility_score, 4),
                    overall_score=round(overall_score, 4),
                )
            )

        screener = sorted(
            screener,
            key=lambda item: (item.buy_probability * 0.45 + item.expected_return * 2.2 + item.factor_score * 0.35),
            reverse=True,
        )[:8]
        factors = sorted(factors, key=lambda item: item.overall_score, reverse=True)[:8]

        sectors: list[SectorRotation] = []
        for sector, items in grouped.items():
            avg_buy = float(np.mean([item.buy_probability for item in items]))
            avg_return = float(np.mean([item.expected_return for item in items]))
            avg_factor = float(np.mean([item.factor_score for item in items]))
            score = avg_buy * 0.45 + avg_return * 2.2 + avg_factor * 0.35
            if score >= 0.62:
                stance = "overweight"
            elif score <= 0.48:
                stance = "underweight"
            else:
                stance = "market_weight"
            sectors.append(
                SectorRotation(
                    sector=sector,
                    average_buy_probability=round(avg_buy, 4),
                    average_expected_return=round(avg_return, 4),
                    average_factor_score=round(avg_factor, 4),
                    stance=stance,
                )
            )
        sectors = sorted(sectors, key=lambda item: item.average_factor_score, reverse=True)

        return screener, factors, sectors

    def _terminal_opportunity_tape(
        self,
        rows: list[PositionView],
        regime_label: str,
    ) -> tuple[list[SignalResponse], list[ForecastResponse]]:
        current_weights = self._current_weights()
        candidates = self._build_barbell_candidates(rows, current_weights)
        price_map = {row.symbol: row.market_price for row in rows}
        selected = sorted(
            candidates,
            key=lambda candidate: (
                candidate.buy_probability * 0.50
                + candidate.expected_return * 2.20
                + candidate.confidence_score * 0.30
            ),
            reverse=True,
        )[:4]

        signals: list[SignalResponse] = []
        forecasts: list[ForecastResponse] = []
        for candidate in selected:
            reference_price = float(price_map.get(candidate.symbol, 100.0))
            band = max(0.04, min(candidate.volatility_score * 0.08, 0.12))
            signals.append(
                SignalResponse(
                    symbol=candidate.symbol,
                    buy_probability=round(candidate.buy_probability, 4),
                    sell_probability=round(max(0.0, 1 - candidate.buy_probability), 4),
                    volatility_forecast=round(max(0.10, min(0.65, 1 - candidate.volatility_score)), 4),
                    confidence_score=round(candidate.confidence_score, 4),
                    market_regime=regime_label,
                )
            )
            forecasts.append(
                ForecastResponse(
                    symbol=candidate.symbol,
                    price_target=round(reference_price * (1 + candidate.expected_return), 2),
                    expected_return=round(candidate.expected_return, 4),
                    confidence_interval_low=round(reference_price * (1 + candidate.expected_return - band), 2),
                    confidence_interval_high=round(reference_price * (1 + candidate.expected_return + band), 2),
                )
            )
        return signals, forecasts

    @staticmethod
    def _barbell_universe_catalog() -> list[dict[str, str]]:
        return [
            {
                "symbol": "SGOV",
                "bucket": "defensive",
                "role": "cash_surrogate",
                "rationale": "Ultra short-duration Treasuries stabilize the book and preserve optionality.",
                "buy_probability": 0.56,
                "expected_return": 0.02,
                "confidence_score": 0.78,
                "quality_score": 0.88,
                "volatility_score": 0.94,
                "liquidity_score": 0.93,
            },
            {
                "symbol": "TLT",
                "bucket": "defensive",
                "role": "duration_hedge",
                "rationale": "Long-duration Treasuries hedge growth and liquidity shocks.",
                "buy_probability": 0.53,
                "expected_return": 0.03,
                "confidence_score": 0.70,
                "quality_score": 0.76,
                "volatility_score": 0.75,
                "liquidity_score": 0.89,
            },
            {
                "symbol": "GLD",
                "bucket": "defensive",
                "role": "real_asset_hedge",
                "rationale": "Gold diversifies inflation, policy, and confidence shocks.",
                "buy_probability": 0.55,
                "expected_return": 0.04,
                "confidence_score": 0.72,
                "quality_score": 0.72,
                "volatility_score": 0.70,
                "liquidity_score": 0.88,
            },
            {
                "symbol": "LLY",
                "bucket": "defensive",
                "role": "quality_defensive_equity",
                "rationale": "Healthcare quality anchors the defensive side with resilient earnings.",
                "buy_probability": 0.61,
                "expected_return": 0.07,
                "confidence_score": 0.75,
                "quality_score": 0.84,
                "volatility_score": 0.74,
                "liquidity_score": 0.82,
            },
            {
                "symbol": "XOM",
                "bucket": "defensive",
                "role": "inflation_hedge",
                "rationale": "Energy exposure offsets inflation and commodity-driven stress regimes.",
                "buy_probability": 0.58,
                "expected_return": 0.06,
                "confidence_score": 0.68,
                "quality_score": 0.73,
                "volatility_score": 0.62,
                "liquidity_score": 0.86,
            },
            {
                "symbol": "JPM",
                "bucket": "defensive",
                "role": "quality_financial",
                "rationale": "A capitalized financial franchise adds carry without pure duration risk.",
                "buy_probability": 0.57,
                "expected_return": 0.05,
                "confidence_score": 0.67,
                "quality_score": 0.74,
                "volatility_score": 0.68,
                "liquidity_score": 0.84,
            },
            {
                "symbol": "MSFT",
                "bucket": "opportunistic",
                "role": "compounder",
                "rationale": "High-quality platform compounding on the upside sleeve.",
                "buy_probability": 0.67,
                "expected_return": 0.09,
                "confidence_score": 0.80,
                "quality_score": 0.89,
                "volatility_score": 0.69,
                "liquidity_score": 0.92,
            },
            {
                "symbol": "AAPL",
                "bucket": "opportunistic",
                "role": "franchise_growth",
                "rationale": "Mega-cap ecosystem exposure retains growth optionality with liquidity.",
                "buy_probability": 0.62,
                "expected_return": 0.07,
                "confidence_score": 0.76,
                "quality_score": 0.86,
                "volatility_score": 0.72,
                "liquidity_score": 0.91,
            },
            {
                "symbol": "NVDA",
                "bucket": "opportunistic",
                "role": "high_beta_growth",
                "rationale": "AI and semiconductor beta sit on the convex upside side of the barbell.",
                "buy_probability": 0.71,
                "expected_return": 0.14,
                "confidence_score": 0.82,
                "quality_score": 0.84,
                "volatility_score": 0.58,
                "liquidity_score": 0.90,
            },
            {
                "symbol": "AMZN",
                "bucket": "opportunistic",
                "role": "consumer_cloud_growth",
                "rationale": "Cloud and consumer platform exposure offer asymmetric upside participation.",
                "buy_probability": 0.64,
                "expected_return": 0.08,
                "confidence_score": 0.74,
                "quality_score": 0.75,
                "volatility_score": 0.64,
                "liquidity_score": 0.88,
            },
            {
                "symbol": "META",
                "bucket": "opportunistic",
                "role": "advertising_ai_growth",
                "rationale": "Advertising cash flow funds upside exposure to AI and platform re-rating.",
                "buy_probability": 0.63,
                "expected_return": 0.08,
                "confidence_score": 0.73,
                "quality_score": 0.72,
                "volatility_score": 0.61,
                "liquidity_score": 0.87,
            },
            {
                "symbol": "QQQ",
                "bucket": "opportunistic",
                "role": "liquid_growth_beta",
                "rationale": "Liquid technology beta keeps the upside sleeve deployable at scale.",
                "buy_probability": 0.60,
                "expected_return": 0.06,
                "confidence_score": 0.70,
                "quality_score": 0.69,
                "volatility_score": 0.67,
                "liquidity_score": 0.94,
            },
        ]

    def _infer_barbell_bucket(self, symbol: str) -> str:
        sector = self._symbol_sector_map().get(symbol.upper(), "default")
        if sector in {"technology", "semiconductors", "communication", "consumer"}:
            return "opportunistic"
        return "defensive"

    @staticmethod
    def _barbell_rationale(
        *,
        regime: str,
        confidence: float,
        target_buckets: dict[str, float],
    ) -> str:
        if regime == "risk_on":
            posture = "tilts capital toward convex upside while preserving a shock absorber"
        elif regime in {"risk_off", "defensive"}:
            posture = "leans on ballast and liquidity before redeploying into selective upside"
        else:
            posture = "keeps capital balanced between resilience and selective growth"
        return (
            f"Barbell posture is {posture}; regime={regime}, confidence={confidence:.0%}, "
            f"defensive={target_buckets['defensive']:.0%}, opportunistic={target_buckets['opportunistic']:.0%}, "
            f"cash={target_buckets['cash']:.0%}."
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
        symbol = Symbol(
            ticker=ticker,
            asset_class="equity",
            exchange="SMART",
            market_data_ticker=ticker,
            market_data_enabled=True,
            currency="USD",
        )
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

        snapshot = quant_insights_service.get_live_market_snapshot_for_requests(
            {
                position.symbol.ticker: self._market_data_symbol(position.symbol)
                for position in positions
            }
        )
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
        outbox_entry = EventOutbox(
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
        try:
            with self.db.begin_nested():
                self.db.add(outbox_entry)
                self.db.flush()
        except SQLAlchemyError as exc:
            if not settings.outbox_fail_open:
                raise
            logger.warning(
                "Outbox persistence skipped for %s due to schema/runtime mismatch: %s",
                event.event_name,
                exc,
            )
            return

        try:
            event_bus.publish(event)
        except Exception as exc:  # pragma: no cover - local handlers are optional
            logger.warning("In-process event bus handler failed for %s: %s", event.event_name, exc)

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

    @staticmethod
    def _market_data_symbol(symbol: Symbol) -> str | None:
        if not symbol.market_data_enabled:
            return None
        return symbol.market_data_ticker or symbol.ticker

    @staticmethod
    def _effective_market_data_ticker(symbol: Symbol) -> str | None:
        if not symbol.market_data_enabled:
            return None
        return symbol.market_data_ticker or symbol.ticker

    @staticmethod
    def _symbol_sector_map() -> dict[str, str]:
        return {
            "AAPL": "technology",
            "MSFT": "technology",
            "NVDA": "semiconductors",
            "AMZN": "consumer",
            "META": "communication",
            "JPM": "financials",
            "XOM": "energy",
            "LLY": "healthcare",
            "SGOV": "bonds",
            "TLT": "bonds",
            "GLD": "commodities",
            "QQQ": "technology",
        }

    @staticmethod
    def _scenario_catalog() -> list[dict[str, object]]:
        return [
            {
                "scenario_id": "1929",
                "name": "Great Depression",
                "period": "1929-1932",
                "trigger": "Credit bubble unwind, bank failures, and forced deleveraging.",
                "summary": "Extreme equity collapse with severe liquidity contraction and deflationary pressure.",
                "macro_context": {
                    "Inflation": "-2.1%",
                    "Unemployment": "24.9%",
                    "Policy rate": "0.6%",
                    "Long bond yield": "2.8%",
                },
                "default_shock": -0.44,
                "cash_buffer_shock": 0.0,
                "sector_shocks": {"technology": -0.49, "financials": -0.58, "energy": -0.31},
                "shock_factors": [("Equities", -0.44), ("Liquidity", -0.18), ("Deflation", -0.07)],
            },
            {
                "scenario_id": "1973_oil",
                "name": "Oil Shock",
                "period": "1973-1974",
                "trigger": "Oil embargo, inflation shock, and recessionary repricing.",
                "summary": "Energy dislocation, margin compression, and growth multiple reset.",
                "macro_context": {
                    "Inflation": "8.7%",
                    "Unemployment": "4.9%",
                    "Policy rate": "10.8%",
                    "Oil price change": "+70%",
                },
                "default_shock": -0.22,
                "cash_buffer_shock": 0.0,
                "sector_shocks": {"technology": -0.25, "energy": 0.08, "financials": -0.16},
                "shock_factors": [("Oil", 0.70), ("Rates", 0.03), ("Equities", -0.22)],
            },
            {
                "scenario_id": "1989",
                "name": "1989 Leverage Crack",
                "period": "1989-1990",
                "trigger": "Leverage unwind, property stress, and global growth slowdown.",
                "summary": "Crowded quality-growth names derate while liquidity thins across cyclicals.",
                "macro_context": {
                    "Inflation": "4.8%",
                    "Unemployment": "5.3%",
                    "Policy rate": "8.1%",
                    "10Y yield": "8.5%",
                },
                "default_shock": -0.16,
                "cash_buffer_shock": 0.0,
                "sector_shocks": {"technology": -0.18, "financials": -0.19, "energy": -0.12},
                "shock_factors": [("Equities", -0.16), ("Credit", -0.05), ("Property", -0.08)],
            },
            {
                "scenario_id": "2000_tech",
                "name": "Tech Bubble Burst",
                "period": "2000-2002",
                "trigger": "Multiple compression in high-duration growth and speculative tech.",
                "summary": "Technology leaders face severe repricing while cash becomes the portfolio stabilizer.",
                "macro_context": {
                    "Inflation": "3.4%",
                    "Unemployment": "4.0%",
                    "Policy rate": "6.5%",
                    "Nasdaq drawdown": "-78%",
                },
                "default_shock": -0.20,
                "cash_buffer_shock": 0.0,
                "sector_shocks": {"technology": -0.34, "semiconductors": -0.39, "communication": -0.26},
                "shock_factors": [("Growth", -0.34), ("Volatility", 0.38), ("Funding", -0.06)],
            },
            {
                "scenario_id": "2008",
                "name": "Global Financial Crisis",
                "period": "2008-2009",
                "trigger": "Housing/credit collapse and systemic bank deleveraging.",
                "summary": "Broad equity drawdown, credit widening, and elevated volatility hit risky assets simultaneously.",
                "macro_context": {
                    "Inflation": "3.8%",
                    "Unemployment": "7.3%",
                    "Policy rate": "2.0%",
                    "Credit spread": "+300bps",
                },
                "default_shock": -0.24,
                "cash_buffer_shock": 0.0,
                "sector_shocks": {"technology": -0.21, "financials": -0.38, "energy": -0.28},
                "shock_factors": [("Equities", -0.24), ("Credit Spread", 0.03), ("Volatility", 0.45)],
            },
            {
                "scenario_id": "2020_pandemic",
                "name": "Pandemic Shock",
                "period": "2020 Q1-Q2",
                "trigger": "Global shutdowns, mobility collapse, and liquidity scramble.",
                "summary": "Fast drawdown with violent policy response and factor dispersion across the book.",
                "macro_context": {
                    "Inflation": "1.2%",
                    "Unemployment": "14.7%",
                    "Policy rate": "0.25%",
                    "GDP shock": "-31.4% annualized",
                },
                "default_shock": -0.13,
                "cash_buffer_shock": 0.0,
                "sector_shocks": {"technology": -0.09, "energy": -0.31, "financials": -0.17},
                "shock_factors": [("Equities", -0.13), ("Rates", -0.015), ("USD", 0.06)],
            },
            {
                "scenario_id": "2022_inflation",
                "name": "Inflation and Rates Shock",
                "period": "2022",
                "trigger": "Sticky inflation, aggressive central-bank hikes, and valuation reset.",
                "summary": "Duration-sensitive equities and growth franchises compress while energy and cash partly cushion.",
                "macro_context": {
                    "Inflation": "8.0%",
                    "Unemployment": "3.6%",
                    "Policy rate": "4.5%",
                    "10Y yield": "4.2%",
                },
                "default_shock": -0.14,
                "cash_buffer_shock": 0.0,
                "sector_shocks": {"technology": -0.21, "energy": 0.12, "financials": -0.09},
                "shock_factors": [("Rates", 0.02), ("Inflation", 0.08), ("Growth", -0.12)],
            },
        ]

    def _build_scenario_result(
        self,
        config: dict[str, object],
        overview: PortfolioOverview,
        rows: list[PositionView],
    ) -> ScenarioResult:
        default_shock = float(config["default_shock"])
        cash_buffer_shock = float(config.get("cash_buffer_shock", 0.0))
        sector_shocks = dict(config.get("sector_shocks", {}))
        macro_context = [
            ScenarioMacroMetric(label=label, value=value)
            for label, value in dict(config.get("macro_context", {})).items()
        ]

        pnl_total = 0.0
        portfolio_impacts: list[ScenarioImpactItem] = []
        shocks: list[ScenarioShock] = []

        for factor_name, shock_value in list(config.get("shock_factors", [])):
            shocks.append(
                ScenarioShock(
                    factor=factor_name,
                    shock=float(shock_value),
                    contribution=0.0,
                )
            )

        for row in rows:
            sector = self._symbol_sector_map().get(row.symbol.upper(), "default")
            position_shock = float(sector_shocks.get(sector, default_shock))
            pnl_impact = round(row.market_value * position_shock, 2)
            pnl_total += pnl_impact
            portfolio_impacts.append(
                ScenarioImpactItem(
                    bucket=row.symbol,
                    pnl_impact=pnl_impact,
                    comment=f"{sector.replace('_', ' ').title()} bucket repriced by {position_shock:.0%}.",
                )
            )

        cash_impact = round(overview.cash * cash_buffer_shock, 2)
        if overview.cash:
            portfolio_impacts.append(
                ScenarioImpactItem(
                    bucket="Cash Buffer",
                    pnl_impact=cash_impact,
                    comment="Cash acts as a liquidity reserve and partial shock absorber.",
                )
            )
            pnl_total += cash_impact

        if shocks and pnl_total != 0:
            equal_contribution = round(pnl_total / len(shocks), 2)
            shocks = [
                ScenarioShock(factor=shock.factor, shock=shock.shock, contribution=equal_contribution)
                for shock in shocks
            ]

        nav = max(overview.nav, 1.0)
        drawdown_impact = round(pnl_total / nav, 4)

        return ScenarioResult(
            scenario_id=str(config["scenario_id"]),
            name=str(config["name"]),
            period=str(config.get("period") or ""),
            trigger=str(config.get("trigger") or ""),
            summary=str(config.get("summary") or ""),
            estimated_pnl_impact=round(pnl_total, 2),
            drawdown_impact=drawdown_impact,
            macro_context=macro_context,
            portfolio_impacts=portfolio_impacts,
            shocks=shocks,
        )
