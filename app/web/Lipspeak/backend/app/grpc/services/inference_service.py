"""inference_service.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: Inference abstraction layer.
@date: 11 June 2026
@returns: Model inference operations.

"""


from app.grpc.clients.grid_client import (
    GridClient,
)

from app.grpc.clients.avsr_client import (
    AVSRClient,
)


class InferenceService:
    """Inference service."""

    @staticmethod
    def predict_video(
        video_bytes: bytes,
    ) -> str:
        """Predict transcript from video."""

        return (
            GridClient.predict(
                video_bytes,
            )
        )

    @staticmethod
    def get_realtime_stub():
        """Get AVSR streaming stub."""

        return (
            AVSRClient.get_stub()
        )