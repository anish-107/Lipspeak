"""auth_service.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Authentication business logic.
@date: 11 June 2026
@returns: Authentication service.

"""


# Imports
from sqlalchemy.orm import Session

from app.models.user import User

from app.core.security import (
    hash_password,
    verify_password,
)


# Auth Service
class AuthService:
    """Authentication service."""

    @staticmethod
    def register(
        db: Session,
        username: str,
        name: str,
        password: str,
    ) -> User:
        """Register new user."""

        existing_user = (
            db.query(User)
            .filter(
                User.username
                == username,
            )
            .first()
        )

        if existing_user:
            raise ValueError(
                "Username already exists.",
            )

        user = User(
            username=username,
            name=name,
            password_hash=
            hash_password(password),
        )

        db.add(user)

        db.commit()

        db.refresh(user)

        return user

    @staticmethod
    def login(
        db: Session,
        username: str,
        password: str,
    ) -> User | None:
        """Authenticate user."""

        user = (
            db.query(User)
            .filter(
                User.username
                == username,
            )
            .first()
        )

        if not user:
            return None

        if not verify_password(
            password,
            user.password_hash,
        ):
            return None

        return user