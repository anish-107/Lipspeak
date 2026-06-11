"""user.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: User ORM model.
@date: 11 June 2026
@returns: User database model.

"""


# Imports
import uuid

from datetime import datetime

from sqlalchemy import (
    String,
    DateTime,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.core.database import (
    Base,
)


# User Model
class User(Base):
    """User model."""

    __tablename__ = "users"

    id: Mapped[str] = (
        mapped_column(
            String(36),
            primary_key=True,
            default=lambda: str(
                uuid.uuid4(),
            ),
        )
    )

    username: Mapped[str] = (
        mapped_column(
            String(100),
            unique=True,
            nullable=False,
        )
    )

    name: Mapped[str] = (
        mapped_column(
            String(255),
            nullable=False,
        )
    )

    password_hash: Mapped[str] = (
        mapped_column(
            String(255),
            nullable=False,
        )
    )

    created_at: Mapped[datetime] = (
        mapped_column(
            DateTime,
            default=datetime.utcnow,
        )
    )

    videos = relationship(
        "Video",
        back_populates="user",
    )