import cv2
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.config import settings

# 1. Initialize Face Landmarker (New API)
model_path: str = str(settings.face_landmarker_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, # Use IMAGE mode for single frames
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# EXACT MediaPipe landmarks for the outer and inner lips
MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 
    269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 
    82, 13, 312, 311, 310, 415
]

def get_mouth_roi(frame):
    # Convert OpenCV BGR to MediaPipe Image object
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 2. Perform Detection (New API)
    results = detector.detect(mp_image)

    # Results are now in a FaceLandmarkerResult object
    if not results.face_landmarks:
        return None

    # Access the first face's landmarks
    landmarks = results.face_landmarks[0]
    h, w, _ = frame.shape

    # Extract coordinates 
    xs = [int(landmarks[i].x * w) for i in MOUTH_LANDMARKS]
    ys = [int(landmarks[i].y * h) for i in MOUTH_LANDMARKS]

    # Reverted padding back to 30 to match Dlib scale
    pad = 30 
    min_x = max(0, min(xs) - pad)
    max_x = min(w, max(xs) + pad)
    min_y = max(0, min(ys) - pad)
    max_y = min(h, max(ys) + pad)

    return min_y, max_y, min_x, max_x

def load_video(path):
    cap = cv2.VideoCapture(path)
    raw_frames = []
    
    # Removed the frame_skip logic. We need every frame for lip reading!
    while True:
        ret, frame = cap.read()
        if not ret: break
        raw_frames.append(frame)
    cap.release()

    if len(raw_frames) < 10:
        raise ValueError("Video is too short or unreadable.")

    mid_idx = len(raw_frames) // 2
    roi = get_mouth_roi(raw_frames[mid_idx])

    if roi is None:
        roi = get_mouth_roi(raw_frames[0])
    if roi is None:
        raise ValueError("No face/mouth detected in the video.")

    min_y, max_y, min_x, max_x = roi
    processed_frames = []

    for frame in raw_frames:
        if max_y > frame.shape[0] or max_x > frame.shape[1]:
            continue

        mouth = frame[min_y:max_y, min_x:max_x]
        mouth = cv2.resize(mouth, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))
        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        mouth = tf.expand_dims(mouth, -1)
        processed_frames.append(mouth)

    if not processed_frames:
        raise ValueError("Could not extract valid mouth frames.")

    frames_tensor = tf.stack(processed_frames)
    mean = tf.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(tf.cast(frames_tensor, tf.float32))
    frames_tensor = tf.cast(frames_tensor - mean, tf.float32) / (std + 1e-6)

    curr_len = frames_tensor.shape[0]
    if curr_len > settings.MAX_FRAMES:
        frames_tensor = frames_tensor[:settings.MAX_FRAMES]
    elif curr_len < settings.MAX_FRAMES:
        # Made sure CONSTANT is explicitly defined, same as Dlib code
        paddings = tf.constant([[0, settings.MAX_FRAMES - curr_len], [0, 0], [0, 0], [0, 0]])
        frames_tensor = tf.pad(frames_tensor, paddings, "CONSTANT")

    return frames_tensor