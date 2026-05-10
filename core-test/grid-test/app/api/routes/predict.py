''' predict.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: This file defines the prediction API routes for the GRID lip reading model.
@date: 10 May 2026
@returns: Prediction API Endpoints

'''


# Imports
import time
from pathlib import (
    Path
)
from datetime import (
    UTC,
    datetime
)
from typing import (
    Annotated
)
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile
)
from app.config import (
    settings
)
from app.api.dependencies import (
    validate_video_file
)
from app.schemas.response import (
    PredictionResponse
)
from app.services.inference_service import (
    inference_service
)



# Router Setup
router = APIRouter()



# Validated Video File Dependency
ValidatedVideoFile = Annotated[
    UploadFile,
    Depends(validate_video_file)
]



# Prediction Endpoint
@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary=(
        "Generate transcript "
        "from uploaded lip reading video"
    )
)
async def predict_video(
    file: ValidatedVideoFile = File(...)
) -> PredictionResponse:

    '''
    @description: Accepts a video file
    and generates transcript predictions
    using the GRID lip reading model.

    @args:
        file:
            Uploaded input video.

    @returns:
        Predicted transcript response.

    '''

    # Initialize Processing Timer
    start_time: float = time.time()


    # Uploaded File Path
    file_path: Path | None = None


    try:

        # Create Upload Directory
        settings.upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # Uploaded Filename
        filename: str = (
            file.filename or "upload.mp4"
        )


        # Uploaded Video Path
        file_path = (
            settings.upload_dir /
            filename
        )


        # Save Uploaded Video
        with open(
            file_path,
            "wb"
        ) as buffer:

            content: bytes = (
                await file.read()
            )

            buffer.write(
                content
            )


        # Generate Prediction
        transcript: str = (
            inference_service.predict(
                str(file_path)
            )
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) from e


    finally:

        # Remove Uploaded File
        if (
            file_path is not None
            and
            file_path.exists()
        ):

            file_path.unlink()


    # Calculate Processing Time
    processing_time: float = round(
        time.time() - start_time,
        4
    )


    # Return Prediction Response
    return PredictionResponse(
        transcript=transcript,
        processing_time_seconds=(
            processing_time
        ),
        timestamp=datetime.now(
            UTC
        )
    )