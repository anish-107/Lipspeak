''' test_model.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Unit tests for TensorFlow model loading and inference.
@date: 10 May 2026
@returns: Model Test Suite

'''


# Imports
import keras
import tensorflow as tf
from app.utils.model_loader import (
    load_model
)



# Test Model Loading
def test_model_loading() -> None:

    '''
    @description: Tests TensorFlow
    model loading.

    '''

    model: keras.Model = (
        load_model()
    )


    assert model is not None

    assert isinstance(
        model,
        keras.Model
    )



# Test Model Inference
def test_model_inference() -> None:

    '''
    @description: Tests TensorFlow
    model inference.

    '''

    model: keras.Model = (
        load_model()
    )


    # Dummy Input Tensor
    dummy_input: tf.Tensor = (
        tf.zeros(
            (
                1,
                75,
                46,
                140,
                1
            )
        )
    )


    # Model Prediction
    output: tf.Tensor = model(
        dummy_input,
        training=False
    )


    assert output is not None

    assert isinstance(
        output,
        tf.Tensor
    )