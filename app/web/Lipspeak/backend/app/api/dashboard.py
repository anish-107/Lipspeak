"""dashboard.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Dashboard endpoints.
@date: 11 June 2026
@returns: Dashboard router.
"""

# Imports
from fastapi import (
    APIRouter,
    Depends,
)
from sqlalchemy.orm import (
    Session,
)
from app.models.user import (
    User,
)
from app.core.database import (
    get_db,
)
from app.api.dependencies import (
    get_current_user,
)
from app.schemas.dashboard import (
    DashboardOverviewResponse,
)
from app.services.dashboard_service import (
    DashboardService,
)

# Router
router = APIRouter(
    prefix="/api/dashboard",
    tags=["Dashboard"],
)

# Overview
@router.get(
    "/overview",
    response_model=DashboardOverviewResponse,
)
def get_dashboard_overview(
    user: User = Depends(
        get_current_user,
    ),
    db: Session = Depends(
        get_db,
    ),
):
    """Get dashboard overview."""
    return (
        DashboardService.get_overview(
            db=db,
            user_id=user.id,
        )
    )