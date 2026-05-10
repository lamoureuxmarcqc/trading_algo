from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.deps import get_platform_service
from app.events.dispatcher import outbox_dispatcher
from app.schemas.admin import (
    AdminUser,
    AuditLogEntry,
    DomainEventDispatchResponse,
    DomainEventEntry,
    DomainEventSummary,
)
from app.services.platform_service import PlatformService

router = APIRouter()


@router.get("/users", response_model=list[AdminUser])
def list_users(platform_service: PlatformService = Depends(get_platform_service)) -> list[AdminUser]:
    return platform_service.list_admin_users()


@router.get("/audit", response_model=list[AuditLogEntry])
def list_audit_logs(
    limit: int = 25,
    platform_service: PlatformService = Depends(get_platform_service),
) -> list[AuditLogEntry]:
    return platform_service.list_audit_logs(limit=limit)


@router.get("/events", response_model=list[DomainEventEntry])
def list_domain_events(
    limit: int = 25,
    platform_service: PlatformService = Depends(get_platform_service),
) -> list[DomainEventEntry]:
    return platform_service.list_domain_events(limit=limit)


@router.get("/events/summary", response_model=DomainEventSummary)
def get_domain_event_summary(
    platform_service: PlatformService = Depends(get_platform_service),
) -> DomainEventSummary:
    return platform_service.summarize_domain_events()


@router.post("/events/dispatch", response_model=DomainEventDispatchResponse)
def dispatch_domain_events(
    background_tasks: BackgroundTasks,
    synchronous: bool = True,
) -> DomainEventDispatchResponse:
    if synchronous:
        result = outbox_dispatcher.dispatch_pending()
        return DomainEventDispatchResponse(
            scanned=result.scanned,
            delivered=result.delivered,
            failed=result.failed,
        )

    background_tasks.add_task(outbox_dispatcher.dispatch_pending)
    return DomainEventDispatchResponse(scanned=0, delivered=0, failed=0)
