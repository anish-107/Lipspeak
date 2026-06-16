''' inference_service.py
@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: Service responsible for
loading and running the Chaplin AVSR
inference pipeline.
@date: 11 June 2026
@returns: Transcript generated from
input video.

'''


# Imports
from pipelines.pipeline import (
    InferencePipeline,
)



# Inference Service
class InferenceService:

    '''
    @description: Handles AVSR model
    loading and inference.

    '''

    def __init__(
        self,
    ):

        '''
        @description: Initializes and
        loads the AVSR inference
        pipeline.

        '''

        print(
            "\nLoading AVSR Model..."
        )

        self.model = (
            InferencePipeline(
                config_filename=(
                    "configs/"
                    "LRS3_V_WER19.1.ini"
                ),
                detector="retinaface",
                face_track=True,
                device="cpu",
            )
        )

        print(
            "AVSR Model Loaded.\n"
        )


    def predict(
        self,
        video_path: str,
    ) -> str:

        '''
        @description: Runs AVSR
        inference on a video file.

        @args:
            video_path:
                Path to video file.

        @returns:
            Predicted transcript.

        '''

        return self.model(
            video_path
        )