from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    success: bool
    transcription: str
    filename: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str