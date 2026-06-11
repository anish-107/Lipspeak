"""auth.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Authentication endpoints.
@date: 11 June 2026
@returns: Authentication router.

"""


# Imports
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from sqlalchemy.orm import Session

from app.models.user import User

from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    AuthResponse,
)

from app.schemas.user import (
    UserResponse,
)

from app.services.auth_service import (
    AuthService,
)

from app.core.security import (
    create_access_token,
)

from app.core.database import (
    get_db,
)

from app.api.dependencies import (
    get_current_user,
)


# Router
router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


# Register
@router.post(
    "/register",
    response_model=UserResponse,
)
def register(
    data: RegisterRequest,
    db: Session = Depends(
        get_db,
    ),
):
    """Register user."""

    try:
        return AuthService.register(
            db=db,
            username=data.username,
            name=data.name,
            password=data.password,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# Login
@router.post(
    "/login",
    response_model=AuthResponse,
)
def login(
    data: LoginRequest,
    db: Session = Depends(
        get_db,
    ),
):
    """Login user."""

    user = AuthService.login(
        db=db,
        username=data.username,
        password=data.password,
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials.",
        )

    token = create_access_token(
        user.id,
    )

    return AuthResponse(
        access_token=token,
        token_type="bearer",
    )


# Current User
@router.get(
    "/me",
    response_model=UserResponse,
)
def current_user(
    user: User = Depends(
        get_current_user,
    ),
):
    """Get current user."""

    return user