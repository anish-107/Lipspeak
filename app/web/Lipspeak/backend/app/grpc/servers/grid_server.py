"""grid_server.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: GRID gRPC server.
@date: 11 June 2026
@returns: Video transcript predictions.

"""

# pyright: reportAttributeAccessIssue=false

from concurrent import (
    futures,
)

import grpc

from app.grpc.protos.lipspeak.v1 import (
    inference_pb2,
    inference_pb2_grpc,
)

from app.ml.predictors.grid_predictor import (
    GridPredictor,
)


class InferenceService(
    inference_pb2_grpc.InferenceServiceServicer
):
    """GRID inference service."""

    def PredictVideo(
        self,
        request,
        context,
    ):
        transcript = (
            GridPredictor.predict(
                request.video,
            )
        )

        return (
            inference_pb2
            .PredictVideoResponse(
                transcript=transcript,
            )
        )


def serve():
    """Start server."""

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=10,
        )
    )

    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceService(),
        server,
    )

    server.add_insecure_port(
        "[::]:50051",
    )

    server.start()

    print(
        "GRID Server running on 50051",
    )

    server.wait_for_termination()


if __name__ == "__main__":
    serve()