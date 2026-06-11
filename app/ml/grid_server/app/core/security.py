''' security.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Security utilities and validation
settings for GRID lip reading backend.
@date: 10 May 2026
@returns: Security Utilities

'''


# Imports
from pathlib import (
    Path
)

from fastapi import (
    HTTPException,
    UploadFile
)



# Allowed Video Extensions
ALLOWED_VIDEO_EXTENSIONS: set[str] = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}



# Maximum Upload Size
MAX_UPLOAD_SIZE: int = (
    50 * 1024 * 1024
)



# Validate Video Extension
def validate_video_extension(
    filename: str
) -> None:

    '''
    @description: Validates uploaded
    video file extension.

    @args:
        filename:
            Uploaded filename.

    '''

    # File Extension
    extension: str = (
        Path(filename)
        .suffix
        .lower()
    )


    # Validate Extension
    if extension not in (
        ALLOWED_VIDEO_EXTENSIONS
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format."
            )
        )



# Validate Video Size
async def validate_video_size(
    file: UploadFile
) -> None:

    '''
    @description: Validates uploaded
    video file size.

    @args:
        file:
            Uploaded video file.

    '''

    # Read File Content
    content: bytes = await file.read()


    # Reset File Pointer
    await file.seek(0)


    # Validate File Size
    if len(content) > (
        MAX_UPLOAD_SIZE
    ):

        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded video exceeds "
                "maximum allowed size."
            )
        )