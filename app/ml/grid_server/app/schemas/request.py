''' request.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: This file defines the request schema for the GRID lip reading model.
@date: 10 May 2026
@returns: Request Schema

'''

# Imports
from typing import Optional
from fastapi import File, UploadFile
from pydantic import BaseModel


# Video Upload Request Schema
class VideoUploadRequest(BaseModel):
    """
    Request schema for optional metadata related to
    uploaded video files.

    Attributes:
        filename (Optional[str]): Name of the uploaded video file.
    """

    filename: Optional[str] = None


# Video File Handler
def video_file(file: UploadFile = File(...)) -> UploadFile:
    """
    Handles incoming uploaded video files.

    Args:
        file (UploadFile): Uploaded video file received
        from multipart/form-data.

    Returns:
        UploadFile: Uploaded video file object.
    """

    return file