"""video.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Video ORM model.
@date: 11 June 2026
@returns: Video database model.

"""


# Imports
import uuid

from datetime import datetime

from sqlalchemy import (
    String,
    DateTime,
    ForeignKey,
    Text,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.core.database import (
    Base,
)


# Video Model
class Video(Base):
    """Video model."""

    __tablename__ = "videos"

    id: Mapped[str] = (
        mapped_column(
            String(36),
            primary_key=True,
            default=lambda: str(
                uuid.uuid4(),
            ),
        )
    )

    user_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            ForeignKey(
                "users.id",
            ),
            nullable=True,
        )
    )

    video_link: Mapped[str] = (
        mapped_column(
            Text,
            nullable=False,
        )
    )

    transcript: Mapped[str] = (
        mapped_column(
            Text,
            nullable=False,
        )
    )

    source_type: Mapped[str] = (
        mapped_column(
            String(50),
            nullable=False,
        )
    )

    created_at: Mapped[datetime] = (
        mapped_column(
            DateTime,
            default=datetime.utcnow,
        )
    )

    user = relationship(
        "User",
        back_populates="videos",
    )