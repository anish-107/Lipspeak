"""videos.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Video upload and retrieval endpoints.
@date: 11 June 2026
@returns: Video router.

"""

# Imports

import os
import time
import tempfile
from app.services.s3_service import (
    S3Service,
)

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from sqlalchemy.orm import (
    Session,
)

from app.api.dependencies import (
    get_current_user,
)

from app.core.database import (
    get_db,
)

from app.models.user import (
    User,
)

from app.schemas.video import (
    UploadResponse,
    VideoResponse,
)


from app.services.video_service import (
    VideoService,
)

from app.services.transcription_service import (
    TranscriptionService,
)

# Router

router = APIRouter(
    prefix="/api/videos",
    tags=["Videos"],
)

# Upload Video

@router.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_video(
    file: UploadFile = File(...),
    user: User = Depends(
        get_current_user,
    ),
    db: Session = Depends(
        get_db,
    ),
):
    """Upload pre-recorded video."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    contents = await file.read()

    request_start = (
    
        time.perf_counter()
    
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4",
    ) as temp_file:

        temp_file.write(
            contents,
        )

        temp_path = (
            temp_file.name
        )

    try:

        s3_start = (
            time.perf_counter()
        )
        
        video_link = (
            S3Service.upload_video(
                temp_path,
                file.filename,
            )
        )
        
        s3_end = (
            time.perf_counter()
        )

        transcription_start = (
            time.perf_counter()
        )
        
        transcript = (
            TranscriptionService
            .transcribe_video(
                contents,
            )
        )
        
        transcription_end = (
            time.perf_counter()
        )

        db_start = (
            time.perf_counter()
        )
        
        video = (
            VideoService.create_video(
                db=db,
                user_id=user.id,
                filename=file.filename,
                video_link=video_link,
                transcript=transcript,
            )
        )
        
        db_end = (
            time.perf_counter()
        )

        request_end = (
            time.perf_counter()
        )
        
        print(
            "\n"
            "========== PIPELINE TIMING ==========\n"
            f"S3 Upload      : "
            f"{s3_end - s3_start:.2f}s\n"
            f"Transcription  : "
            f"{transcription_end - transcription_start:.2f}s\n"
            f"Database Save  : "
            f"{db_end - db_start:.2f}s\n"
            f"Total Request  : "
            f"{request_end - request_start:.2f}s\n"
            "=====================================\n"
        )

        return UploadResponse(
            transcript=transcript,
            video_id=video.id,
        )

    finally:

        if os.path.exists(
            temp_path,
        ):
            os.remove(
                temp_path,
            )


# Get All Videos

@router.get(
    "",
    response_model=list[VideoResponse],
)
def get_videos(
    user: User = Depends(
        get_current_user,
    ),
    db: Session = Depends(
        get_db,
    ),
):
    """Get user videos."""

    return (
        VideoService.get_videos(
            db=db,
            user_id=user.id,
        )
    )


# Get Video By ID

@router.get(
    "/{video_id}",
    response_model=VideoResponse,
)
def get_video(
    video_id: str,
    user: User = Depends(
        get_current_user,
    ),
    db: Session = Depends(
        get_db,
    ),
):
    """Get video details."""

    video = (
        VideoService.get_video(
            db=db,
            video_id=video_id,
        )
    )

    if not video:
        raise HTTPException(
            status_code=404,
            detail="Video not found.",
        )

    video.video_link = (
        S3Service
        .generate_presigned_url(
            video.video_link,
        )
    )
    
    return video