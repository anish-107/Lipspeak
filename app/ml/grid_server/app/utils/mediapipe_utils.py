''' mediapipe_utils.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Utility functions for MediaPipe face landmark detection.
@date: 10 May 2026
@returns: Mouth ROI Utilities

'''


# Imports
from typing import Optional
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.config import settings



# MediaPipe Setup
base_options = python.BaseOptions(
    model_asset_path=str(
        settings.face_landmarker_path
    )
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(
    options
)



# Mouth Landmarks
MOUTH_LANDMARKS: list[int] = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267,
    269, 270, 409, 78, 95,
    88, 178, 87, 14, 317,
    402, 318, 324, 308, 191,
    80, 81, 82, 13, 312,
    311, 310, 415
]



# Get Mouth ROI
def get_mouth_roi(
    frame
) -> Optional[
    tuple[int, int, int, int]
]:
    """
    Extracts mouth region coordinates
    using MediaPipe face landmarks.

    Args:
        frame:
            Input video frame.

    Returns:
        Optional[
            tuple[int, int, int, int]
        ]:
            Mouth ROI coordinates as:
            (
                min_y,
                max_y,
                min_x,
                max_x
            )

            Returns None if no face
            is detected.
    """

    # Convert frame from BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Detect face landmarks
    results = detector.detect(
        mp_image
    )


    # Return None if no face detected
    if not results.face_landmarks:
        return None


    # Extract first detected face
    landmarks = results.face_landmarks[0]


    # Frame dimensions
    h: int
    w: int

    h, w, _ = frame.shape


    # Extract mouth x coordinates
    xs: list[int] = [
        int(landmarks[i].x * w)
        for i in MOUTH_LANDMARKS
    ]


    # Extract mouth y coordinates
    ys: list[int] = [
        int(landmarks[i].y * h)
        for i in MOUTH_LANDMARKS
    ]


    # Mouth bounding box padding
    pad: int = 30


    # Mouth ROI Coordinates
    min_x: int = max(
        0,
        min(xs) - pad
    )

    max_x: int = min(
        w,
        max(xs) + pad
    )

    min_y: int = max(
        0,
        min(ys) - pad
    )

    max_y: int = min(
        h,
        max(ys) + pad
    )


    return (
        min_y,
        max_y,
        min_x,
        max_x
    )