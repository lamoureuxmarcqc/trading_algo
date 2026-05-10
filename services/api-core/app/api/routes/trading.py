from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status

from app.api.deps import get_platform_service
from app.core.config import settings
from app.events.dispatcher import outbox_dispatcher
from app.schemas.trading import Fill, OrderCreate, OrderResponse
from app.services.platform_service import IdempotencyConflictError, PlatformService

router = APIRouter()


@router.post("", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
def create_order(
    payload: OrderCreate,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    platform_service: PlatformService = Depends(get_platform_service),
) -> OrderResponse:
    try:
        result = platform_service.create_order(
            payload,
            idempotency_key=getattr(request.state, "idempotency_key", None),
            tenant_id=getattr(request.state, "tenant_id", "public"),
            correlation_id=getattr(request.state, "correlation_id", None),
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    response.status_code = status.HTTP_200_OK if result.replayed else status.HTTP_201_CREATED
    response.headers["X-Idempotency-Status"] = "replayed" if result.replayed else "created"
    if settings.outbox_dispatch_after_write and not result.replayed:
        background_tasks.add_task(outbox_dispatcher.dispatch_pending, 1)
    return result.order


@router.get("", response_model=list[OrderResponse])
def list_orders(platform_service: PlatformService = Depends(get_platform_service)) -> list[OrderResponse]:
    return platform_service.list_orders()


@router.delete("/{order_id}")
def cancel_order(
    order_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    platform_service: PlatformService = Depends(get_platform_service),
) -> dict[str, str]:
    if not platform_service.cancel_order(
        order_id,
        tenant_id=getattr(request.state, "tenant_id", "public"),
        correlation_id=getattr(request.state, "correlation_id", None),
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    if settings.outbox_dispatch_after_write:
        background_tasks.add_task(outbox_dispatcher.dispatch_pending, 1)
    return {"status": "cancelled"}


@router.get("/fills", response_model=list[Fill])
def list_fills(platform_service: PlatformService = Depends(get_platform_service)) -> list[Fill]:
    return platform_service.list_fills()
