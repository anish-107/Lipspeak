"""grid_client.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: GRID model gRPC client.
@date: 11 June 2026
@returns: GRID transcript response.

"""

# pyright: reportAttributeAccessIssue=false

import grpc
import time
from app.grpc.protos.lipspeak.v1 import (
    inference_pb2,
    inference_pb2_grpc,
)


class GridClient:
    """GRID gRPC client."""

    @staticmethod
    def predict(
        video_bytes: bytes,
    ) -> str:

        grpc_start = (
            time.perf_counter()
        )
        
        channel = grpc.insecure_channel(
            "localhost:50051",
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
            f"Backend → GRID RPC: "
            f"{grpc_end - grpc_start:.2f}s"
        )
        
        return (
            response.transcript
        )