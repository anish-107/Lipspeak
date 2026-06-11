''' inference_service.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Service functions for
GRID lip reading inference.
@date: 10 May 2026
@returns: Lip Reading Predictions

'''


# Imports
import keras
import time
import tensorflow as tf

from typing import cast

from app.utils.model_loader import (
    load_model
)

from app.services.preprocessing_service import (
    load_video
)

from app.services.decoding_service import (
    decode_prediction
)

from app.services.postprocessing_service import (
    postprocess_prediction
)



# Inference Service
class InferenceService:

    '''
    @description: Handles GRID lip reading
    inference pipeline.

    '''


    # Initialize Service
    def __init__(
        self
    ) -> None:

        '''
        @description: Initializes
        inference service.

        '''

        self.model: keras.Model = (
            load_model()
        )


    # Predict Video
    def predict(
        self,
        video_path: str
    ) -> str:

        '''
        @description: Performs lip reading
        prediction on input video.

        @args:
            video_path:
                Input video path.

        @returns:
            Predicted sentence.

        '''

        pipeline_start = (
            time.perf_counter()
        )

        # Load Video Tensor
        load_start = (
            time.perf_counter()
        )
        
        video_tensor: tf.Tensor = (
            load_video(
                video_path
            )
        )
        
        load_end = (
            time.perf_counter()
        )


        # Add Batch Dimension
        expand_start = (
            time.perf_counter()
        )
        
        video_tensor = tf.expand_dims(
            video_tensor,
            axis=0
        )
        
        expand_end = (
            time.perf_counter()
        )


        # Predict Output
        predict_start = (
            time.perf_counter()
        )
        
        yhat: tf.Tensor = cast(
            tf.Tensor,
            self.model.predict(
                video_tensor
            )
        )
        
        predict_end = (
            time.perf_counter()
        )


        # Decode Prediction
        decode_start = (
            time.perf_counter()
        )
        
        prediction: str = (
            decode_prediction(
                yhat
            )
        )
        
        decode_end = (
            time.perf_counter()
        )

        # Postprocess Prediction
        post_start = (
            time.perf_counter()
        )
        
        prediction = (
            postprocess_prediction(
                prediction
            )
        )
        
        post_end = (
            time.perf_counter()
        )


        pipeline_end = (
            time.perf_counter()
        )
        
        print(
            "\n"
            "===== MODEL PROFILE =====\n"
            f"Video Load      : "
            f"{load_end - load_start:.2f}s\n"
            f"Expand Dim      : "
            f"{expand_end - expand_start:.4f}s\n"
            f"TF Predict      : "
            f"{predict_end - predict_start:.2f}s\n"
            f"CTC Decode      : "
            f"{decode_end - decode_start:.2f}s\n"
            f"Postprocess     : "
            f"{post_end - post_start:.2f}s\n"
            f"TOTAL MODEL     : "
            f"{pipeline_end - pipeline_start:.2f}s\n"
            "=========================\n"
        )

        return prediction



# Global Inference Service
inference_service = (
    InferenceService()
)