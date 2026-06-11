"""realtime_transcription_service.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: Realtime transcription orchestration.
@date: 11 June 2026
@returns: Realtime transcript generation.

"""


from app.grpc.services.inference_service import (
    InferenceService,
)


class RealtimeTranscriptionService:
    """Realtime transcription service."""

    def __init__(self):
        """Initialize service."""

        self.partial_transcript = ""

        self.stub = (
            InferenceService
            .get_realtime_stub()
        )

    def process_chunk(
        self,
        chunk: bytes,
    ) -> str:
        """Process incoming chunk."""

        transcript = (
            f"Received {len(chunk)} bytes"
        )

        self.partial_transcript = (
            transcript
        )

        return (
            self.partial_transcript
        )

    def reset(self):
        """Reset session."""

        self.partial_transcript = ""