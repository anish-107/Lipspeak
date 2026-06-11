"""avsr_server.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: AVSR gRPC server.
@date: 11 June 2026
@returns: Streaming transcript predictions.

"""


# pyright: reportAttributeAccessIssue=false

from concurrent import futures

import grpc

from app.grpc.protos.lipspeak.v1 import (
    inference_pb2,
    inference_pb2_grpc,
)

from app.ml.predictors.avsr_predictor import (
    AVSRPredictor,
)


class InferenceService(
    inference_pb2_grpc
    .InferenceServiceServicer
):

    def PredictRealtime(
        self,
        request_iterator,
        context,
    ):

        for chunk in request_iterator:

            transcript = (
                AVSRPredictor.predict(
                    chunk.chunk,
                )
            )

            yield (
                inference_pb2
                .PredictRealtimeResponse(
                    transcript=transcript,
                )
            )


def serve():

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=10,
        )
    )

    (
        inference_pb2_grpc
        .add_InferenceServiceServicer_to_server(
            InferenceService(),
            server,
        )
    )

    server.add_insecure_port(
        "[::]:50052",
    )

    server.start()

    print(
        "AVSR Server running on 50052",
    )

    server.wait_for_termination()


if __name__ == "__main__":
    serve()