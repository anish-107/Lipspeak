''' test_api.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Unit tests for API endpoints.
@date: 10 May 2026
@returns: API Test Suite

'''


# Imports
from fastapi.testclient import (
    TestClient
)
from app.main import (
    app
)



# Test Client
client = TestClient(
    app
)



# Test Root Endpoint
def test_root() -> None:

    '''
    @description: Tests root endpoint.

    '''

    response = client.get(
        "/"
    )


    assert response.status_code == 200

    assert (
        response.json()["message"]
        ==
        "GRID Lip Reading API Running"
    )



# Test Health Endpoint
def test_health() -> None:

    '''
    @description: Tests health endpoint.

    '''

    response = client.get(
        "/health"
    )


    assert response.status_code == 200