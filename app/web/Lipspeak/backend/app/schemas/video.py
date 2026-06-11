"""video.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Video schemas.
@date: 11 June 2026
@returns: Video request and response schemas.

"""


# Imports
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
)


# Video Response
class VideoResponse(
    BaseModel,
):
    """Video response schema."""

    id: str

    original_filename: str

    video_link: str

    transcript: str

    source_type: str

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


# Upload Response
class UploadResponse(
    BaseModel,
):
    """Upload response schema."""

    transcript: str

    video_id: str | None