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
            max_workers=10,
        )
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
        "\nAuto-AVSR gRPC Server started on port 50052\n"
    )

    server.wait_for_termination()


if __name__ == "__main__":
    serve()