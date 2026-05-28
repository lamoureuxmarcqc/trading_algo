from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from starlette.background import BackgroundTasks
from starlette.responses import Response


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.api.routes import trading  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.models import Account, AuditLog, EventOutbox, IdempotencyKey, Order, User  # noqa: E402
from app.events.bus import event_bus  # noqa: E402
from app.events.contracts import DomainEvent  # noqa: E402
from app.events.dispatcher import OutboxDispatcher  # noqa: E402
from app.schemas.portfolio import BarbellStrategyConfig  # noqa: E402
from app.schemas.trading import OrderCreate, OrderResponse  # noqa: E402
from app.services.platform_service import (  # noqa: E402
    IdempotencyConflictError,
    OrderSubmissionResult,
    PlatformService,
)


def _build_service() -> tuple[Session, PlatformService]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(bind=engine)
    session.add(
        User(
            email=settings.default_user_email,
            full_name="Default CIO",
            password_hash="not-used-in-this-test",
            is_active=True,
        )
    )
    session.add(
        Account(
            broker_name="Interactive Brokers",
            account_number="TEST-001",
            account_type="family_office",
            base_currency="USD",
            status="active",
        )
    )
    session.commit()
    return session, PlatformService(session)


def _payload(quantity: float = 100.0) -> OrderCreate:
    return OrderCreate(
        symbol="MSFT",
        side="BUY",
        order_type="limit",
        quantity=quantity,
        limit_price=415.2,
        broker="paper",
        strategy_tag="test",
    )


def test_create_order_replays_same_resource_for_same_idempotency_key() -> None:
    session, service = _build_service()

    first = service.create_order(_payload(), idempotency_key="idem-001", tenant_id="tenant-alpha")
    second = service.create_order(_payload(), idempotency_key="idem-001", tenant_id="tenant-alpha")

    assert first.replayed is False
    assert second.replayed is True
    assert first.order.id == second.order.id
    assert len(session.scalars(select(Order)).all()) == 1
    assert len(session.scalars(select(IdempotencyKey)).all()) == 1
    assert len(session.scalars(select(EventOutbox)).all()) == 1
    assert len(session.scalars(select(AuditLog)).all()) == 1


def test_create_order_rejects_payload_mismatch_for_same_idempotency_key() -> None:
    session, service = _build_service()
    service.create_order(_payload(quantity=100.0), idempotency_key="idem-002", tenant_id="tenant-alpha")

    with pytest.raises(IdempotencyConflictError):
        service.create_order(_payload(quantity=250.0), idempotency_key="idem-002", tenant_id="tenant-alpha")

    assert len(session.scalars(select(Order)).all()) == 1
    assert len(session.scalars(select(IdempotencyKey)).all()) == 1


def test_create_order_scopes_idempotency_by_tenant() -> None:
    session, service = _build_service()

    first = service.create_order(_payload(), idempotency_key="idem-003", tenant_id="tenant-alpha")
    second = service.create_order(_payload(), idempotency_key="idem-003", tenant_id="tenant-beta")

    assert first.order.id != second.order.id
    assert len(session.scalars(select(Order)).all()) == 2
    assert len(session.scalars(select(IdempotencyKey)).all()) == 2
    assert len(session.scalars(select(EventOutbox)).all()) == 2


def test_create_order_records_domain_event_and_notifies_local_bus() -> None:
    session, service = _build_service()
    captured_events = []
    event_bus._subscribers.clear()
    event_bus.subscribe("com.terminal.orders.created.v1", captured_events.append)

    result = service.create_order(
        _payload(),
        idempotency_key="idem-100",
        tenant_id="tenant-alpha",
        correlation_id="corr-abc",
    )

    outbox_events = session.scalars(select(EventOutbox)).all()
    assert result.replayed is False
    assert len(outbox_events) == 1
    assert outbox_events[0].event_name == "com.terminal.orders.created.v1"
    assert outbox_events[0].topic == "terminal.orders.v1"
    assert outbox_events[0].tenant_id == "tenant-alpha"
    assert outbox_events[0].correlation_id == "corr-abc"
    assert outbox_events[0].payload["order_id"] == result.order.id
    assert len(captured_events) == 1
    assert captured_events[0].aggregate_id == result.order.id


def test_portfolio_refresh_records_domain_event() -> None:
    session, service = _build_service()

    result = service.refresh_portfolio_market_data(
        force=False,
        tenant_id="tenant-alpha",
        correlation_id="corr-refresh",
    )

    outbox_events = session.scalars(select(EventOutbox)).all()
    assert result.status == "refreshed"
    assert len(outbox_events) == 1
    assert outbox_events[0].event_name == "com.terminal.portfolio.refreshed.v1"
    assert outbox_events[0].topic == "terminal.portfolio.v1"
    assert outbox_events[0].tenant_id == "tenant-alpha"
    assert outbox_events[0].correlation_id == "corr-refresh"


def test_cancel_order_records_domain_event() -> None:
    session, service = _build_service()
    created = service.create_order(_payload(), tenant_id="tenant-alpha")

    cancelled = service.cancel_order(
        created.order.id,
        tenant_id="tenant-alpha",
        correlation_id="corr-cancel",
    )

    outbox_events = session.scalars(
        select(EventOutbox).order_by(EventOutbox.created_at.asc())
    ).all()
    assert cancelled is True
    assert len(outbox_events) == 2
    assert outbox_events[1].event_name == "com.terminal.orders.cancelled.v1"
    assert outbox_events[1].correlation_id == "corr-cancel"


def test_dispatcher_marks_pending_event_as_delivered() -> None:
    session, service = _build_service()
    service.create_order(_payload(), tenant_id="tenant-alpha", correlation_id="corr-dispatch")
    factory = sessionmaker(bind=session.get_bind())
    dispatched_ids: list[str] = []

    class FakeSink:
        def publish(self, event: EventOutbox) -> None:
            dispatched_ids.append(event.id)

    dispatcher = OutboxDispatcher(session_factory=factory, sink=FakeSink())
    result = dispatcher.dispatch_pending()

    session.expire_all()
    refreshed = session.scalars(select(EventOutbox)).all()
    assert result.scanned == 1
    assert result.delivered == 1
    assert result.failed == 0
    assert len(dispatched_ids) == 1
    assert refreshed[0].delivery_status == "delivered"
    assert refreshed[0].attempt_count == 1
    assert refreshed[0].dispatched_at is not None
    assert refreshed[0].last_error is None


def test_dispatcher_marks_failed_event_and_keeps_it_retryable() -> None:
    session, service = _build_service()
    service.create_order(_payload(), tenant_id="tenant-alpha")
    factory = sessionmaker(bind=session.get_bind())

    class FailingSink:
        def publish(self, event: EventOutbox) -> None:
            raise RuntimeError(f"boom:{event.event_name}")

    dispatcher = OutboxDispatcher(session_factory=factory, sink=FailingSink())
    result = dispatcher.dispatch_pending()

    session.expire_all()
    refreshed = session.scalars(select(EventOutbox)).all()
    assert result.scanned == 1
    assert result.delivered == 0
    assert result.failed == 1
    assert refreshed[0].delivery_status == "failed"
    assert refreshed[0].attempt_count == 1
    assert refreshed[0].dispatched_at is None
    assert refreshed[0].last_error == "boom:com.terminal.orders.created.v1"


def test_domain_event_summary_counts_statuses() -> None:
    session, service = _build_service()
    service.create_order(_payload(), tenant_id="tenant-alpha")
    events = session.scalars(select(EventOutbox)).all()
    events[0].delivery_status = "failed"
    session.commit()

    summary = service.summarize_domain_events()

    assert summary.failed == 1
    assert summary.pending == 0
    assert summary.delivered == 0


def test_publish_domain_event_fails_open_when_outbox_schema_is_not_ready(monkeypatch) -> None:
    session, service = _build_service()
    event = DomainEvent(
        event_name="com.terminal.portfolio.refreshed.v1",
        topic="terminal.portfolio.v1",
        aggregate_type="portfolio",
        aggregate_id="family-office-master",
        payload={"status": "refreshed"},
    )

    def broken_flush() -> None:
        raise SQLAlchemyError("missing outbox column")

    original_flush = service.db.flush
    monkeypatch.setattr(service.db, "flush", broken_flush)

    service._publish_domain_event(event)
    monkeypatch.setattr(service.db, "flush", original_flush)

    assert len(session.scalars(select(EventOutbox)).all()) == 0


def test_list_scenarios_exposes_historical_stress_catalog() -> None:
    _session, service = _build_service()

    scenarios = service.list_scenarios()
    scenario_ids = {scenario.scenario_id for scenario in scenarios}

    assert "1929" in scenario_ids
    assert "1973_oil" in scenario_ids
    assert "1989" in scenario_ids
    assert "2000_tech" in scenario_ids
    assert "2008" in scenario_ids
    assert "2020_pandemic" in scenario_ids
    assert "2022_inflation" in scenario_ids
    assert any(scenario.macro_context for scenario in scenarios)


def test_portfolio_correlation_matrix_returns_square_matrix() -> None:
    _session, service = _build_service()

    matrix = service.get_portfolio_correlation_matrix()

    assert len(matrix.symbols) == len(matrix.matrix)
    assert all(len(row) == len(matrix.symbols) for row in matrix.matrix)
    for index, row in enumerate(matrix.matrix):
        assert row[index] == 1.0


def test_barbell_allocation_returns_balanced_targets() -> None:
    _session, service = _build_service()

    allocation = service.get_barbell_allocation()
    total_target = sum(item.target_weight for item in allocation.allocations)

    assert allocation.regime in {"risk_on", "neutral", "risk_off", "defensive"}
    assert pytest.approx(total_target, abs=0.0005) == 1.0
    assert any(item.bucket == "defensive" for item in allocation.allocations)
    assert any(item.bucket == "opportunistic" for item in allocation.allocations)
    assert any(item.symbol == "CASH" for item in allocation.allocations)
    assert allocation.rebalance_instructions


def test_barbell_allocation_accepts_custom_bucket_targets() -> None:
    _session, service = _build_service()

    allocation = service.get_barbell_allocation(
        BarbellStrategyConfig(
            defensive_weight_target=0.50,
            opportunistic_weight_target=0.30,
            cash_buffer_target=0.20,
            max_positions_per_bucket=2,
        )
    )

    assert allocation.defensive_weight == pytest.approx(0.5, abs=0.0001)
    assert allocation.opportunistic_weight == pytest.approx(0.3, abs=0.0001)
    assert allocation.cash_buffer_weight == pytest.approx(0.2, abs=0.0001)
    assert len([item for item in allocation.allocations if item.bucket == "defensive"]) <= 2
    assert len([item for item in allocation.allocations if item.bucket == "opportunistic"]) <= 2


def test_terminal_snapshot_aggregates_terminal_sections() -> None:
    _session, service = _build_service()

    snapshot = service.get_terminal_snapshot()

    assert snapshot.portfolio.portfolio_id == "family-office-master"
    assert snapshot.signals
    assert snapshot.forecasts
    assert snapshot.scenarios
    assert snapshot.barbell.allocations
    assert snapshot.event_summary.delivered >= 0
    assert snapshot.users
    assert snapshot.audit_logs is not None


def test_trading_route_sets_replay_status_headers() -> None:
    request = SimpleNamespace(state=SimpleNamespace(idempotency_key="idem-004", tenant_id="tenant-alpha"))
    response = Response()
    background_tasks = BackgroundTasks()

    class FakePlatformService:
        def create_order(self, *_args, **_kwargs) -> OrderSubmissionResult:
            return OrderSubmissionResult(
                order=OrderResponse(
                    id="order-1",
                    symbol="MSFT",
                    side="BUY",
                    status="accepted",
                    order_type="limit",
                    quantity=100.0,
                    filled_quantity=0.0,
                    limit_price=415.2,
                    stop_price=None,
                    broker="paper",
                    created_at="2026-05-08T00:00:00+00:00",
                ),
                replayed=True,
            )

    order = trading.create_order(_payload(), request, response, background_tasks, FakePlatformService())

    assert order.id == "order-1"
    assert response.status_code == 200
    assert response.headers["X-Idempotency-Status"] == "replayed"


def test_trading_route_maps_idempotency_conflict_to_http_409() -> None:
    request = SimpleNamespace(state=SimpleNamespace(idempotency_key="idem-005", tenant_id="tenant-alpha"))
    response = Response()
    background_tasks = BackgroundTasks()

    class FakePlatformService:
        def create_order(self, *_args, **_kwargs) -> OrderSubmissionResult:
            raise IdempotencyConflictError("conflicting request")

    with pytest.raises(trading.HTTPException) as exc_info:
        trading.create_order(_payload(), request, response, background_tasks, FakePlatformService())

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "conflicting request"
