import cv2
import dlib # pyright: ignore
import tensorflow as tf
from app.config import settings


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(settings.dlib_landmark_path))

MOUTH_POINTS = list(range(48, 61))


def get_mouth_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    landmarks = predictor(gray, faces[0])

    xs = [landmarks.part(i).x for i in MOUTH_POINTS]
    ys = [landmarks.part(i).y for i in MOUTH_POINTS]

    pad = 30
    min_x = max(0, min(xs) - pad)
    max_x = min(frame.shape[1], max(xs) + pad)
    min_y = max(0, min(ys) - pad)
    max_y = min(frame.shape[0], max(ys) + pad)

    return min_y, max_y, min_x, max_x


def load_video(path):
    cap = cv2.VideoCapture(path)
    raw_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
        paddings = tf.constant([
            [0, settings.MAX_FRAMES - curr_len],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        frames_tensor = tf.pad(frames_tensor, paddings, "CONSTANT")

    return frames_tensor