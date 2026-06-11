"""
demo.py
@description: Anonymous transcription endpoint.
"""

from fastapi import (
    APIRouter,
    File,
    UploadFile,
)

from app.services.transcription_service import (
    TranscriptionService,
)

router = APIRouter(
    prefix="/api/demo",
    tags=["Demo"],
)


@router.post(
    "/transcribe",
)
async def transcribe_demo(
    file: UploadFile = File(...),
):
    contents = await file.read()

    transcript = (
        TranscriptionService
        .transcribe_video(
            contents,
        )
    )

    return {
        "transcript":
        transcript,
    }