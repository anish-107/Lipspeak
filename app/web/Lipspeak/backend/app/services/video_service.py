"""video_service.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Video business logic.
@date: 11 June 2026
@returns: Video operations.

"""


# Imports
from sqlalchemy.orm import Session

from app.models.video import Video


# Video Service
class VideoService:
    """Video service."""

    @staticmethod
    def create_video(
        db: Session,
        user_id: str,
        filename: str,
        video_link: str,
        transcript: str,
    ) -> Video:
        """Create video record."""

        video = Video(
            user_id=user_id,
            original_filename=
            filename,
            video_link=
            video_link,
            transcript=
            transcript,
            source_type=
            "pre-recorded",
        )

        db.add(video)

        db.commit()

        db.refresh(video)

        return video

    @staticmethod
    def get_videos(
        db: Session,
        user_id: str,
    ):
        """Get user videos."""

        return (
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

    @staticmethod
    def get_video(
        db: Session,
        video_id: str,
    ):
        """Get video."""

        return (
            db.query(Video)
            .filter(
                Video.id ==
                video_id,
            )
            .first()
        )