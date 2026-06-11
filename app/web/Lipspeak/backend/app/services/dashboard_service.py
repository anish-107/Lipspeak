"""dashboard_service.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Dashboard business logic.
@date: 11 June 2026
@returns: Dashboard operations.
"""

# Imports
from sqlalchemy.orm import (
    Session,
)
from app.models.video import (
    Video,
)

# Dashboard Service
class DashboardService:
    """Dashboard service."""
    @staticmethod
    def get_overview(
        db: Session,
        user_id: str,
    ):
        """Get dashboard overview."""

        videos = (
            db.query(Video)
            .filter(
                Video.user_id ==
                user_id,
            )
            .order_by(
                Video.created_at.desc(),
            )
            .all()
        )

        total_videos = len(
            videos,
        )

        latest_transcript = (
            videos[0].transcript
            if videos
            else "No transcripts yet."
        )

        return {
            "total_videos":
            total_videos,

            "latest_transcript":
            latest_transcript,

            "recent_videos":
            videos[:5],
        }