"""security.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Password hashing and JWT utilities.
@date: 11 June 2026
@returns: Security helper functions.

"""


# Imports
from datetime import (
    datetime,
    timedelta,
    UTC,
)

from jose import jwt

from passlib.context import (
    CryptContext,
)

from app.core.config import (
    settings,
)


# Password Context
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


# Hash Password
def hash_password(
    password: str,
) -> str:
    """Hash plain password."""

    return pwd_context.hash(
        password,
    )


# Verify Password
def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """Verify password."""

    return pwd_context.verify(
        plain_password,
        hashed_password,
    )


# Create Access Token
def create_access_token(
    user_id: str,
) -> str:
    """Create JWT token."""

    expire = (
        datetime.now(UTC)
        + timedelta(
            minutes=
            settings.JWT_EXPIRE_MINUTES,
        )
    )

    payload = {
        "sub": user_id,
        "exp": expire,
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=
        settings.JWT_ALGORITHM,
    )