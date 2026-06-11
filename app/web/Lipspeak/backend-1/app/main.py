"""main.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: FastAPI application entry point.
@date: 11 June 2026
@returns: FastAPI application.

"""


# Imports
from fastapi import FastAPI

from app.core.config import (
    settings,
)


# Application
app = FastAPI(
    title=settings.APP_NAME,
)


# Health Check
@app.get("/")
def root():
    """Health check endpoint."""

    return {
        "message": (
            "LipSpeak AI Backend Running"
        )
    }