''' avsr_service.py
@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: gRPC service exposing
AVSR inference functionality.
@date: 11 June 2026
'''

# pyright: reportAttributeAccessIssue=false

import os
import tempfile

from app.services.inference_service import (
    InferenceService,
)

from app.grpc.protos.lipspeak.v1 import (
    inference_pb2,
    inference_pb2_grpc,
)


class AVSRService(
    inference_pb2_grpc.InferenceServiceServicer
):

    def __init__(self):

        self.inference_service = (
            InferenceService()
        )

    def PredictVideo(
        self,
        request,
        context,
    ):

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False,
            ) as temp_file:

                temp_file.write(
                    request.video
                )

                temp_path = (
                    temp_file.name
                )

            transcript = (
                self.inference_service.predict(
                    temp_path
                )
            )
            
            print(
                "\nAVSR TRANSCRIPT:",
                repr(transcript),
                "\n",
            )

            return (
                inference_pb2
                .PredictVideoResponse(
                    transcript=transcript
                )
            )

        finally:

            if (
                temp_path
                and
                os.path.exists(
                    temp_path
                )
            ):
                os.unlink(
                    temp_path
                )