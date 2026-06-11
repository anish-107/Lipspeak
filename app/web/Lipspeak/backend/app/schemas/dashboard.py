"""dashboard.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Dashboard response schemas.
@date: 11 June 2026
@returns: Dashboard schemas.
"""

# Imports
from pydantic import (
    BaseModel,
)
from app.schemas.video import (
    VideoResponse,
)

# Dashboard Response
class DashboardOverviewResponse(
    BaseModel,
):
    """Dashboard overview."""
    total_videos: int

    latest_transcript: str

    recent_videos: list[
        VideoResponse
    ]