''' test_client.py
@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: AVSR gRPC test client.
@date: 11 June 2026
'''

import grpc

from app.grpc.protos.lipspeak.v1 import (
    inference_pb2,
    inference_pb2_grpc,
)


def main():

    with open(
        "sample.mp4",
        "rb",
    ) as file:

        video_bytes = file.read()

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
            channel
        )
    )

    response = (
        stub.PredictVideo(
            inference_pb2
            .PredictVideoRequest(
                video=video_bytes
            )
        )
    )

    print(
        "\nTranscript:"
    )

    print(
        response.transcript
    )


if __name__ == "__main__":
    main()