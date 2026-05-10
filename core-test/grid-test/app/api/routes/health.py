''' health.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: This file defines health check API routes for the GRID lip reading model.
@date: 10 May 2026
@returns: Health Check API Endpoints

'''


# Imports
from datetime import UTC, datetime
from typing import Dict
from fastapi import APIRouter



# Router Setup
router = APIRouter()



# Health Check Endpoint
@router.get(
    "/health",
    summary="Health Check Endpoint"
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint used to verify
    whether the API service is running.

    Returns:
        Dict[str, str]: Health status response.
    """

    return {
        "status": "healthy",
        "service": "GRID Lip Reading API",
        "timestamp": str(datetime.now(UTC))
    }