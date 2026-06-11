"""realtime.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: Real-time websocket endpoint.
@date: 11 June 2026
@returns: Realtime websocket router.

"""


# Imports
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)

from app.services.realtime_transcription_service import (
    RealtimeTranscriptionService,
)


# Router
router = APIRouter()


# Realtime Endpoint
@router.websocket(
    "/ws/realtime",
)
async def realtime_socket(
    websocket: WebSocket,
):
    """Realtime websocket endpoint."""

    await websocket.accept()

    service = (
        RealtimeTranscriptionService()
    )

    try:

        while True:

            message = (
                await websocket.receive()
            )

            # Binary Video Chunk
            if (
                "bytes" in message
                and message["bytes"]
                is not None
            ):

                chunk = (
                    message["bytes"]
                )

                transcript = (
                    service.process_chunk(
                        chunk,
                    )
                )

                await websocket.send_json(
                    {
                        "transcript":
                        transcript,
                    }
                )

            # Text Message
            elif (
                "text" in message
                and message["text"]
                is not None
            ):

                transcript = (
                    service.process_chunk(
                        message[
                            "text"
                        ].encode(),
                    )
                )

                await websocket.send_json(
                    {
                        "transcript":
                        transcript,
                    }
                )

            # Disconnect Event
            elif (
                message["type"]
                == "websocket.disconnect"
            ):
                break

    except WebSocketDisconnect:

        print(
            "Realtime session ended.",
        )

    finally:

        service.reset()

        try:
            await websocket.close()
        except Exception:
            pass