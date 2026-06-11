"""dependencies.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Shared API dependencies.
@date: 11 June 2026
@returns: Authentication dependencies.

"""


# Imports
from jose import jwt

from fastapi import (
    Depends,
    HTTPException,
    status,
)

from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
)

from sqlalchemy.orm import Session

from app.models.user import User

from app.core.config import (
    settings,
)

from app.core.database import (
    get_db,
)


# Security
bearer_scheme = HTTPBearer()


# Current User Dependency
def get_current_user(
    credentials:
    HTTPAuthorizationCredentials = Depends(
        bearer_scheme,
    ),
    db: Session = Depends(
        get_db,
    ),
) -> User:
    """Get authenticated user."""

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET_KEY,
            algorithms=[
                settings.JWT_ALGORITHM,
            ],
        )

        user_id = payload.get(
            "sub",
        )

        if not user_id:
            raise HTTPException(
                status_code=
                status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token.",
            )

        user = (
            db.query(User)
            .filter(
                User.id == user_id,
            )
            .first()
        )

        if not user:
            raise HTTPException(
                status_code=
                status.HTTP_401_UNAUTHORIZED,
                detail="User not found.",
            )

        return user

    except Exception:
        raise HTTPException(
            status_code=
            status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
        )