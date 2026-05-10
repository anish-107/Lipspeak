''' tasks.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Background Celery tasks
for GRID lip reading inference.
@date: 10 May 2026
@returns: Celery Tasks

'''


# Imports
from app.core.logger import (
    logger
)

from app.workers.celery_worker import (
    celery_app
)

from app.services.inference_service import (
    inference_service
)



# Process Video Task
@celery_app.task
def process_video_task(
    video_path: str
) -> str:

    '''
    @description: Processes uploaded
    video asynchronously using
    lip reading inference pipeline.

    @args:
        video_path:
            Uploaded video path.

    @returns:
        Predicted transcript.

    '''

    logger.info(
        "Starting background inference."
    )


    # Generate Prediction
    prediction: str = (
        inference_service.predict(
            video_path
        )
    )


    logger.info(
        "Background inference completed."
    )


    return prediction