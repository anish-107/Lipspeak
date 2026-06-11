"""server.py

@description: GRID gRPC server.

"""


from concurrent import futures

import grpc

from app.grpc.protos.lipspeak.v1 import (
    inference_pb2_grpc,
)

from app.grpc.services.grid_service import (
    GridService,
)


def start_grpc_server():
    """Start GRID gRPC server."""

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=10,
        )
    )

    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        GridService(),
        server,
    )

    server.add_insecure_port(
        "[::]:50051",
    )

    server.start()

    print(
        "GRID gRPC Server running on 50051",
    )

    return server