''' inference_service.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Service functions for
GRID lip reading inference.
@date: 10 May 2026
@returns: Lip Reading Predictions

'''


# Imports
import keras
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

        # Load Video Tensor
        video_tensor: tf.Tensor = (
            load_video(
                video_path
            )
        )


        # Add Batch Dimension
        video_tensor = tf.expand_dims(
            video_tensor,
            axis=0
        )


        # Predict Output
        yhat: tf.Tensor = cast(
            tf.Tensor,
            self.model.predict(
                video_tensor
            )
        )


        # Decode Prediction
        prediction: str = (
            decode_prediction(
                yhat
            )
        )


        # Postprocess Prediction
        prediction = (
            postprocess_prediction(
                prediction
            )
        )


        return prediction



# Global Inference Service
inference_service = (
    InferenceService()
)