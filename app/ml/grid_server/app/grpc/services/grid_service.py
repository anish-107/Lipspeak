"""grid_service.py

@description: GRID gRPC service implementation.

"""


# pyright: reportAttributeAccessIssue=false

import os
import time
import tempfile

from app.grpc.protos.lipspeak.v1 import (
    inference_pb2,
    inference_pb2_grpc,
)

from app.services.inference_service import (
    inference_service,
)


class GridService(
    inference_pb2_grpc.InferenceServiceServicer,
):
    """GRID gRPC service."""

    def PredictVideo(
        self,
        request,
        context,
    ):
        """Predict transcript from video."""
        grpc_start = (
            time.perf_counter()
        )
        
        temp_path = None

        try:
            temp_start = (
                time.perf_counter()
            )
            
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            ) as temp_file:

                temp_file.write(
                    request.video,
                )

                temp_path = (
                    temp_file.name
                )
                
            temp_end = (
                time.perf_counter()
            )
            
            inference_start = (
                time.perf_counter()
            )
            
            transcript = (
                inference_service.predict(
                    temp_path,
                )
            )
            
            inference_end = (
                time.perf_counter()
            )

            grpc_end = (
                time.perf_counter()
            )
            
            print(
                "\n"
                "========== GRID TIMING ==========\n"
                f"Temp File     : "
                f"{temp_end - temp_start:.2f}s\n"
                f"Inference     : "
                f"{inference_end - inference_start:.2f}s\n"
                f"Total gRPC    : "
                f"{grpc_end - grpc_start:.2f}s\n"
                "=================================\n"
            )
            
            return (
                inference_pb2.PredictVideoResponse(
                    transcript=transcript,
                )
            )

        except Exception as e:

            context.set_details(
                str(e),
            )

            return (
                inference_pb2.PredictVideoResponse(
                    transcript="",
                )
            )

        finally:

            if (
                temp_path
                and
                os.path.exists(
                    temp_path,
                )
            ):
                os.remove(
                    temp_path,
                )

    def PredictRealtime(
        self,
        request_iterator,
        context,
    ):
        """
        Placeholder for realtime.
        """

        for _ in request_iterator:

            yield (
                inference_pb2.PredictRealtimeResponse(
                    transcript=(
                        "Realtime not implemented"
                    )
                )
            )