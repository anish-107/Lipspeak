''' preprocessing_service.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Service functions for video preprocessing
using MediaPipe mouth ROI extraction.
@date: 10 May 2026
@returns: Preprocessed Video Tensor

'''


# Imports
import cv2
import tensorflow as tf
from app.utils.mediapipe_utils import (
    get_mouth_roi
)
from app.utils.video_utils import (
    normalize_frames,
    pad_frames,
    process_mouth_frame
)



# Load Video
def load_video(
    video_path: str
) -> tf.Tensor:

    '''
    @description: Loads and preprocesses
    video frames for lip reading inference.

    @args:
        video_path:
            Input video path.

    @returns:
        Preprocessed video tensor.

    '''

    # Open Video Capture
    cap = cv2.VideoCapture(
        video_path
    )


    # Raw Video Frames
    raw_frames: list[
        cv2.typing.MatLike
    ] = []


    # Read Frames
    while True:

        ret: bool
        frame: cv2.typing.MatLike

        ret, frame = cap.read()


        if not ret:
            break


        raw_frames.append(
            frame
        )


    cap.release()


    # Validate Frame Count
    if len(raw_frames) < 10:

        raise ValueError(
            "Video is too short or unreadable."
        )


    # Middle Frame Index
    mid_idx: int = (
        len(raw_frames) // 2
    )


    # Extract Mouth ROI
    roi = get_mouth_roi(
        raw_frames[mid_idx]
    )


    # Fallback First Frame
    if roi is None:

        roi = get_mouth_roi(
            raw_frames[0]
        )


    # Validate ROI
    if roi is None:

        raise ValueError(
            "No face or mouth detected."
        )


    # ROI Coordinates
    min_y: int
    max_y: int
    min_x: int
    max_x: int

    min_y, max_y, min_x, max_x = roi


    # Processed Frames
    processed_frames: list[
        tf.Tensor
    ] = []


    # Process Frames
    for frame in raw_frames:

        # Validate ROI Bounds
        if (
            max_y > frame.shape[0]
            or
            max_x > frame.shape[1]
        ):

            continue


        # Extract Mouth ROI
        mouth = frame[
            min_y:max_y,
            min_x:max_x
        ]


        # Process Mouth Frame
        processed_frame = (
            process_mouth_frame(
                mouth
            )
        )


        processed_frames.append(
            processed_frame
        )


    # Validate Processed Frames
    if not processed_frames:

        raise ValueError(
            "Could not extract mouth frames."
        )


    # Stack Frames
    frames_tensor: tf.Tensor = (
        tf.stack(
            processed_frames
        )
    )


    # Normalize Frames
    frames_tensor = normalize_frames(
        frames_tensor
    )


    # Pad Frames
    frames_tensor = pad_frames(
        frames_tensor
    )


    return frames_tensor