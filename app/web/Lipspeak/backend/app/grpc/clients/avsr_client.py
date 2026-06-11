"""avsr_client.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: AVSR model gRPC client.
@date: 11 June 2026
@returns: AVSR streaming client.

"""


import grpc

from app.grpc.protos.lipspeak.v1 import (
    inference_pb2_grpc,
)


class AVSRClient:
    """AVSR gRPC client."""

    @staticmethod
    def get_stub():

        channel = grpc.insecure_channel(
            "localhost:50052",
        )

        return (
            inference_pb2_grpc
            .InferenceServiceStub(
                channel,
            )
        )