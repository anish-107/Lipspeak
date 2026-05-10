''' test_preprocessing.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Unit tests for video
preprocessing utilities.
@date: 10 May 2026
@returns: Preprocessing Test Suite

'''


# Imports
import tensorflow as tf
from app.utils.video_utils import (
    normalize_frames,
    pad_frames
)



# Test Normalize Frames
def test_normalize_frames() -> None:

    '''
    @description: Tests frame normalization.

    '''

    frames: tf.Tensor = tf.random.normal(
        (
            10,
            46,
            140,
            1
        )
    )


    normalized: tf.Tensor = (
        normalize_frames(
            frames
        )
    )


    assert normalized is not None

    assert isinstance(
        normalized,
        tf.Tensor
    )



# Test Pad Frames
def test_pad_frames() -> None:

    '''
    @description: Tests frame padding.

    '''

    frames: tf.Tensor = tf.random.normal(
        (
            10,
            46,
            140,
            1
        )
    )


    padded: tf.Tensor = (
        pad_frames(
            frames
        )
    )


    assert padded.shape[0] == 75