''' root.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: This file defines the root API route for the GRID lip reading model.
@date: 10 May 2026
@returns: Root API Endpoint

'''



# Imports
from fastapi import APIRouter



# Router Setup
router = APIRouter()



# Root Route
@router.get("/")
async def root() -> dict[str, str]:

    '''
    @description: Root API endpoint.

    @returns:
        API status response.

    '''

    return {
        "message":
        "GRID Lip Reading API Running"
    }