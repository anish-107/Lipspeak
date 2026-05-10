''' dependencies.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: This file defines reusable FastAPI dependencies for the GRID lip reading model.
@date: 10 May 2026
@returns: Shared API Dependencies

'''


# Imports
from typing import Generator
from fastapi import HTTPException, UploadFile



# Validate Uploaded Video File
def validate_video_file(
    file: UploadFile
) -> UploadFile:
    """
    Validates whether the uploaded file
    is a supported video format.

    Args:
        file (UploadFile): Uploaded video file.

    Returns:
        UploadFile: Validated uploaded video file.

    Raises:
        HTTPException: Raised when uploaded file
        is not a valid video.
    """

    # Check if content type is missing
    if file.content_type is None:
        raise HTTPException(
            status_code=400,
            detail="File content type is missing."
        )


    # Check if content type is not a video
    if not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a valid video."
        )


    return file



# Request Timer
def request_timer() -> Generator[None, None, None]:
    """
    Placeholder dependency for request timing,
    logging, or middleware extensions.

    Yields:
        Generator[None, None, None]: Empty generator.
    """


    yield