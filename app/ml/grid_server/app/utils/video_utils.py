''' video_utils.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Utility functions for video preprocessing.
@date: 10 May 2026
@returns: Video Processing Utilities

'''


# Imports
import cv2
import tensorflow as tf
from app.config import settings



# Normalize Frames
def normalize_frames(
    frames_tensor: tf.Tensor
) -> tf.Tensor:

    '''
    @description: Normalizes video frame tensor
    using mean and standard deviation.

    @args:
        frames_tensor:
            Input video frames tensor.

    @returns:
        Normalized video frames tensor.

    '''

    # Frame Mean
    mean: tf.Tensor = tf.reduce_mean(
        frames_tensor
    )


    # Frame Standard Deviation
    std: tf.Tensor = tf.math.reduce_std(
        tf.cast(
            frames_tensor,
            tf.float32
        )
    )


    return tf.cast(
        frames_tensor - mean,
        tf.float32
    ) / (std + 1e-6)



# Pad Frames
def pad_frames(
    frames_tensor: tf.Tensor
) -> tf.Tensor:

    '''
    @description: Pads or truncates video
    frames to fixed sequence length.

    @args:
        frames_tensor:
            Input video frames tensor.

    @returns:
        Padded or truncated video frames tensor.

    '''

    # Current Frame Length
    curr_len_shape: int | None = (
        frames_tensor.shape[0]
    )
    
    curr_len: int = int(
        curr_len_shape or 0
    )


    # Truncate Frames
    if curr_len > settings.MAX_FRAMES:

        return frames_tensor[
            :settings.MAX_FRAMES
        ]


    # Pad Frames
    if curr_len < settings.MAX_FRAMES:

        paddings: tf.Tensor = tf.constant([
            [
                0,
                settings.MAX_FRAMES - curr_len
            ],
            [0, 0],
            [0, 0],
            [0, 0]
        ])


        return tf.pad(
            frames_tensor,
            paddings,
            "CONSTANT"
        )


    return frames_tensor



# Process Mouth Frame
def process_mouth_frame(
    mouth: cv2.typing.MatLike
) -> tf.Tensor:

    '''
    @description: Processes mouth ROI frame
    by resizing and converting to grayscale.

    @args:
        mouth:
            Mouth ROI image frame.

    @returns:
        Processed mouth frame tensor.

    '''

    # Resize Mouth ROI
    mouth = cv2.resize(
        mouth,
        (
            settings.FRAME_WIDTH,
            settings.FRAME_HEIGHT
        )
    )


    # Convert to Grayscale
    mouth = cv2.cvtColor(
        mouth,
        cv2.COLOR_BGR2GRAY
    )


    return tf.expand_dims(
        mouth,
        axis=-1
    )