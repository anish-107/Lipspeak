"""transcription_service.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: Video transcription orchestration service.
@date: 11 June 2026
@returns: Transcript generation services.

"""


from app.grpc.services.inference_service import (
    InferenceService,
)


class TranscriptionService:
    """Transcription service."""

    @staticmethod
    def transcribe_grid(
        video_bytes: bytes,
    ) -> str:

        return (
            InferenceService
            .predict_grid(
                video_bytes,
            )
        )

    @staticmethod
    def transcribe_avsr(
        video_bytes: bytes,
    ) -> str:

        return (
            InferenceService
            .predict_avsr(
                video_bytes,
            )
        )