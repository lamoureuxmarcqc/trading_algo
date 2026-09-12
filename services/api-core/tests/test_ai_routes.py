from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.background import BackgroundTasks
from starlette.responses import Response


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.api.routes import ai as ai_routes  # noqa: E402
from app.schemas.ai import AiPaperStageRequest, AiPaperStageResponse  # noqa: E402
from app.schemas.trading import OrderResponse  # noqa: E402
from app.services.ai_trading_service import PaperTradingDisabledError  # noqa: E402
from app.services.platform_service import (  # noqa: E402
    IdempotencyConflictError,
    PaperStageSubmissionResult,
)


def _order() -> OrderResponse:
    return OrderResponse(
        id="order-ai-1",
        symbol="AAPL",
        side="BUY",
        status="pending",
        order_type="limit",
        quantity=12.5,
        filled_quantity=0.0,
        limit_price=191.2,
        stop_price=175.9,
        broker="paper",
        created_at="2026-09-11T00:00:00+00:00",
    )


def test_stage_ai_paper_orders_requires_idempotency_key() -> None:
    request = SimpleNamespace(state=SimpleNamespace(tenant_id="tenant-alpha"))
    response = Response()

    with pytest.raises(ai_routes.HTTPException) as exc_info:
        ai_routes.stage_ai_paper_orders(
            AiPaperStageRequest(paper_trading_enabled=True),
            request,
            response,
            BackgroundTasks(),
            platform_service=SimpleNamespace(),
            ai_trading_service=SimpleNamespace(),
        )

    assert exc_info.value.status_code == 400
    assert "Idempotency-Key" in exc_info.value.detail


def test_stage_ai_paper_orders_sets_created_status_and_headers() -> None:
    request = SimpleNamespace(
        state=SimpleNamespace(
            idempotency_key="idem-ai-001",
            tenant_id="tenant-alpha",
            correlation_id="corr-ai",
        )
    )
    response = Response()

    class FakeAiTradingService:
        def plan_paper_stage(self, payload: AiPaperStageRequest):
            return (
                [{"symbol": "AAPL", "quantity": 12.5, "stop_price": 175.9}],
                [],
                ["fallback explanation"],
            )

    class FakePlatformService:
        def stage_paper_orders(self, payload: AiPaperStageRequest, candidates, skipped, warnings, **kwargs):
            assert kwargs["idempotency_key"] == "idem-ai-001"
            assert kwargs["tenant_id"] == "tenant-alpha"
            assert kwargs["correlation_id"] == "corr-ai"
            assert candidates[0]["symbol"] == "AAPL"
            return PaperStageSubmissionResult(
                response=AiPaperStageResponse(
                    generated_at="2026-09-11T00:00:00+00:00",
                    policy_id=payload.policy_id,
                    status="staged",
                    replayed=False,
                    orders=[_order()],
                    skipped=skipped,
                    warnings=warnings,
                ),
                replayed=False,
            )

    result = ai_routes.stage_ai_paper_orders(
        AiPaperStageRequest(paper_trading_enabled=True),
        request,
        response,
        BackgroundTasks(),
        platform_service=FakePlatformService(),
        ai_trading_service=FakeAiTradingService(),
    )

    assert result.orders[0].id == "order-ai-1"
    assert response.status_code == 201
    assert response.headers["X-Idempotency-Status"] == "created"


def test_stage_ai_paper_orders_maps_disabled_and_conflict_errors() -> None:
    request = SimpleNamespace(state=SimpleNamespace(idempotency_key="idem-ai-002", tenant_id="tenant-alpha"))
    response = Response()

    class DisabledAiTradingService:
        def plan_paper_stage(self, payload: AiPaperStageRequest):
            raise PaperTradingDisabledError("disabled")

    with pytest.raises(ai_routes.HTTPException) as disabled_exc:
        ai_routes.stage_ai_paper_orders(
            AiPaperStageRequest(),
            request,
            response,
            BackgroundTasks(),
            platform_service=SimpleNamespace(),
            ai_trading_service=DisabledAiTradingService(),
        )

    assert disabled_exc.value.status_code == 403

    class CandidateAiTradingService:
        def plan_paper_stage(self, payload: AiPaperStageRequest):
            return ([{"symbol": "AAPL", "quantity": 12.5, "stop_price": 175.9}], [], [])

    class ConflictPlatformService:
        def stage_paper_orders(self, *_args, **_kwargs):
            raise IdempotencyConflictError("conflict")

    with pytest.raises(ai_routes.HTTPException) as conflict_exc:
        ai_routes.stage_ai_paper_orders(
            AiPaperStageRequest(paper_trading_enabled=True),
            request,
            Response(),
            BackgroundTasks(),
            platform_service=ConflictPlatformService(),
            ai_trading_service=CandidateAiTradingService(),
        )

    assert conflict_exc.value.status_code == 409
    assert conflict_exc.value.detail == "conflict"
