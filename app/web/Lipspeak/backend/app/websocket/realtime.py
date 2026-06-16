"""realtime.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: Real-time websocket endpoint.
@date: 11 June 2026
@returns: Realtime websocket router.

"""

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)

from app.services.realtime_transcription_service import (
    RealtimeTranscriptionService,
)


router = APIRouter()


@router.websocket(
    "/ws/realtime",
)
async def realtime_socket(
    websocket: WebSocket,
):
    """Realtime websocket endpoint."""

    await websocket.accept()

    print(
        "Realtime session started."
    )

    service = (
        RealtimeTranscriptionService()
    )

    try:

        while True:

            message = (
                await websocket.receive()
            )

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

                if transcript:

                    await websocket.send_json(
                        {
                            "transcript":
                            transcript,
                        }
                    )

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

                if transcript:

                    await websocket.send_json(
                        {
                            "transcript":
                            transcript,
                        }
                    )

            elif (
                message["type"]
                == "websocket.disconnect"
            ):
                break

    except WebSocketDisconnect:

        print(
            "Realtime session ended."
        )

    finally:

        service.reset()

        try:

            await websocket.close()
        except Exception:
            pass