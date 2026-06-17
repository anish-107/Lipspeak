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
    inference_pb2_grpc.InferenceServiceServicer,
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

        with tempfile.NamedTemporaryFile(
            suffix=".webm",
            delete=False,
        ) as temp_file:

            temp_file.write(
                request.video
            )

            temp_path = temp_file.name

        try:

            transcript = (
                self.inference_service.predict(
                    temp_path
                )
            )

            print(
                f"\nAUTO_AVSR: {transcript}\n"
            )

            return (
                inference_pb2
                .PredictVideoResponse(
                    transcript=transcript
                )
            )

        finally:

            if os.path.exists(
                temp_path
            ):
                os.remove(
                    temp_path
                )

    def PredictRealtime(
        self,
        request_iterator,
        context,
    ):

        for request in request_iterator:

            with tempfile.NamedTemporaryFile(
                suffix=".webm",
                delete=False,
            ) as temp_file:

                temp_file.write(
                    request.chunk
                )

                temp_path = temp_file.name

            try:

                transcript = (
                    self.inference_service.predict(
                        temp_path
                    )
                )

                yield (
                    inference_pb2
                    .PredictRealtimeResponse(
                        transcript=transcript
                    )
                )

            finally:

                if os.path.exists(
                    temp_path
                ):
                    os.remove(
                        temp_path
                    )