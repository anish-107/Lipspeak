"""inference_service.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: Inference abstraction layer.
@date: 12 June 2026
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
    def predict_grid(
        video_bytes: bytes,
    ) -> str:

        return (
            GridClient.predict(
                video_bytes,
            )
        )

    @staticmethod
    def predict_avsr(
        video_bytes: bytes,
    ) -> str:

        return (
            AVSRClient.predict(
                video_bytes,
            )
        )

    @staticmethod
    def get_realtime_stub():

        return (
            AVSRClient.get_stub()
        )