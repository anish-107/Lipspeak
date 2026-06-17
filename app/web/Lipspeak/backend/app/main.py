"""main.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: FastAPI application entry point.
@date: 11 June 2026
@returns: FastAPI application.

"""


# Imports
from fastapi import FastAPI
import os

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

from app.api.demo import (
    router as demo_router
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# Application
app = FastAPI(
    title=settings.APP_NAME,
    docs_url="/docs" if ENVIRONMENT == "development" else None, 
    redoc_url="/redoc" if ENVIRONMENT == "development" else None, 
    openapi_url="/openapi.json" if ENVIRONMENT == "development" else None
)

origins = [
    "https://lipspeak.anishx.me",  
]


if ENVIRONMENT == "development":
    origins.append("http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],           
    allow_headers=["*"],          
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

app.include_router (
    demo_router,
)

app.include_router(
    dashboard_router,
)

app.include_router(
    realtime_router,
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