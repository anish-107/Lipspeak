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

from app.api.system import (
    router as system_router,
)

from app.api.auth import (
    router as auth_router,
)

from app.api.videos import (
    router as videos_router,
)

from app.api.dashboard import (
    router as dashboard_router,
)

from app.websocket.realtime import (
    router as realtime_router,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

# Application
app = FastAPI(
    title=settings.APP_NAME,
)


app.include_router(
    auth_router,
)

app.include_router(
    system_router,
)

app.include_router(
    videos_router,
)

app.include_router(
    dashboard_router,
)

app.include_router(
    realtime_router,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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