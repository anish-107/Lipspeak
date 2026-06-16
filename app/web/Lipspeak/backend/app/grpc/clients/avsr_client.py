"""avsr_client.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: AVSR model gRPC client.
@date: 12 June 2026
@returns: AVSR transcript response.

"""

# pyright: reportAttributeAccessIssue=false

import grpc
import time

from app.grpc.protos.lipspeak.v1 import (
    inference_pb2,
    inference_pb2_grpc,
)


class AVSRClient:
    """AVSR gRPC client."""

    @staticmethod
    def predict(
        video_bytes: bytes,
    ) -> str:

        grpc_start = (
            time.perf_counter()
        )

        channel = grpc.insecure_channel(
            "localhost:50052",
            options=[
                (
                    "grpc.max_send_message_length",
                    100 * 1024 * 1024,
                ),
                (
                    "grpc.max_receive_message_length",
                    100 * 1024 * 1024,
                ),
            ],
        )

        stub = (
            inference_pb2_grpc
            .InferenceServiceStub(
                channel,
            )
        )

        response = (
            stub.PredictVideo(
                inference_pb2
                .PredictVideoRequest(
                    video=video_bytes,
                )
            )
        )

        grpc_end = (
            time.perf_counter()
        )

        print(
            f"Backend → AVSR RPC: "
            f"{grpc_end - grpc_start:.2f}s"
        )

        return (
            response.transcript
        )

    @staticmethod
    def get_stub():

        channel = grpc.insecure_channel(
            "localhost:50052",
            options=[
                (
                    "grpc.max_send_message_length",
                    100 * 1024 * 1024,
                ),
                (
                    "grpc.max_receive_message_length",
                    100 * 1024 * 1024,
                ),
            ],
        )

        return (
            inference_pb2_grpc
            .InferenceServiceStub(
                channel,
            )
        )