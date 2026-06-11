''' response.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: This file defines the response schema for the GRID lip reading model.
@date: 10 May 2026
@returns: Response Schema

'''


# Imports
from datetime import datetime
from pydantic import BaseModel


# Prediction Response Schema
class PredictionResponse(BaseModel):
    """
    Response schema returned after successful
    lip reading inference.

    Attributes:
        transcript (str): Predicted transcript generated
        from the input video.

        processing_time_seconds (float): Total time taken
        by the model to process the request.

        timestamp (datetime): API response timestamp.
    """

    transcript: str
    processing_time_seconds: float
    timestamp: datetime