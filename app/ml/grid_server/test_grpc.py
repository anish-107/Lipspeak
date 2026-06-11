# pyright: reportAttributeAccessIssue=false

import grpc

from app.grpc.protos.lipspeak.v1 import (
    inference_pb2,
    inference_pb2_grpc,
)


channel = grpc.insecure_channel(
    "localhost:50051",
)

stub = (
    inference_pb2_grpc.InferenceServiceStub(
        channel,
    )
)

with open(
    "sample.mpg",
    "rb",
) as file:

    response = (
        stub.PredictVideo(
            inference_pb2.PredictVideoRequest(
                video=file.read(),
            )
        )
    )

print(
    response.transcript,
)