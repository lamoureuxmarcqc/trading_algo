from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.platform_service import PlatformService


def get_platform_service(db: Session = Depends(get_db)) -> PlatformService:
    return PlatformService(db)
