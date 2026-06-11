"""auth.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Authentication schemas.
@date: 11 June 2026
@returns: Authentication request and response schemas.

"""


# Imports
from pydantic import (
    BaseModel,
)


# Register Request
class RegisterRequest(
    BaseModel,
):
    username: str

    name: str

    password: str


# Login Request
class LoginRequest(
    BaseModel,
):
    username: str

    password: str


# Auth Response
class AuthResponse(
    BaseModel,
):
    access_token: str

    token_type: str