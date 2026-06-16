''' server.py
@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: AVSR gRPC server.
@date: 11 June 2026
'''

from concurrent import futures

import grpc

from app.grpc.services.avsr_service import (
    AVSRService,
)

from app.grpc.protos.lipspeak.v1 import (
    inference_pb2_grpc,
)


def serve():

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=4,
        ),
        options=[
            (
                "grpc.max_receive_message_length",
                100 * 1024 * 1024,
            ),
            (
                "grpc.max_send_message_length",
                100 * 1024 * 1024,
            ),
        ],
    )

    inference_pb2_grpc\
        .add_InferenceServiceServicer_to_server(
            AVSRService(),
            server,
        )

    server.add_insecure_port(
        "[::]:50052"
    )

    server.start()

    print(
        "AVSR gRPC Server started "
        "on port 50052"
    )

    server.wait_for_termination()


if __name__ == "__main__":
    serve()