''' model_loader.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Utility functions for loading TensorFlow lip reading models.
@date: 10 May 2026
@returns: Loaded TensorFlow Model

'''


# Imports
import keras
from typing import cast
import tensorflow as tf
from app.config import settings



# Global Model Instance
model: keras.Model | None = None



# CTC Loss Function
def CTCLoss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor
) -> tf.Tensor:

    '''
    @description: Custom CTC loss function used
    while loading the lip reading model.

    @args:
        y_true:
            Ground truth labels.

        y_pred:
            Predicted logits.

    @returns:
        Computed CTC loss.

    '''

    # Batch Size
    batch_size: tf.Tensor = tf.cast(
        tf.shape(y_true)[0],
        tf.int64
    )


    # Input Sequence Length
    input_len: tf.Tensor = tf.cast(
        tf.shape(y_pred)[1],
        tf.int64
    )


    # Label Sequence Length
    label_len: tf.Tensor = tf.cast(
        tf.shape(y_true)[1],
        tf.int64
    )


    # Input Length Tensor
    input_len = input_len * tf.ones(
        (
            batch_size,
            1
        ),
        dtype=tf.int64
    )


    # Label Length Tensor
    label_len = label_len * tf.ones(
        (
            batch_size,
            1
        ),
        dtype=tf.int64
    )


    return tf.keras.backend.ctc_batch_cost(
        y_true,
        y_pred,
        input_len,
        label_len
    )



# Load TensorFlow Model
def load_model() -> keras.Model:

    '''
    @description: Loads TensorFlow lip reading model.

    @returns:
        Loaded TensorFlow model.

    '''

    global model


    # Return cached model
    if model is not None:
        return model


    # Load TensorFlow model
    model = cast(
        keras.Model,
        keras.models.load_model(
            settings.model_path,
            custom_objects={
                "CTCLoss": CTCLoss
            },
            compile=False
        )
    )


    # Warmup Prediction Input
    dummy_input: tf.Tensor = tf.zeros(
        (
            1,
            settings.MAX_FRAMES,
            settings.FRAME_HEIGHT,
            settings.FRAME_WIDTH,
            1
        )
    )


    # Warmup Prediction
    cast(
        keras.Model,
        model
    ).predict(
        dummy_input,
    )


    return model