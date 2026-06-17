# """
# Auto-AVSR inference service.
# """

# import os
# import cv2
# import torch
# import tempfile
# import numpy as np

# from lightning import ModelModule
# from datamodule.transforms import VideoTransform
# from preparation.detectors.mediapipe.detector import (
#     LandmarksDetector,
# )
# from preparation.detectors.mediapipe.video_process import (
#     VideoProcess,
# )


# class InferenceService:

#     def __init__(self):

#         print(
#             "\nLoading Auto-AVSR Model..."
#         )

#         self.device = torch.device(
#             "cuda"
#             if torch.cuda.is_available()
#             else "cpu"
#         )

#         class Args:
#             modality = "video"

#         self.modelmodule = (
#             ModelModule(
#                 Args(),
#             )
#         )

#         checkpoint = torch.load(
#             "/home/anish/Desktop/Lipspeak/app/ml/auto_avsr/pretrained_models/vsr_trlrs2lrs3vox2avsp_base.pth",
#             map_location=self.device,
#         )

#         self.modelmodule.model.load_state_dict(
#             checkpoint,
#         )

#         self.modelmodule.eval()
#         self.modelmodule.to(
#             self.device,
#         )

#         self.landmarks_detector = (
#             LandmarksDetector()
#         )

#         self.video_process = (
#             VideoProcess(
#                 convert_gray=False,
#             )
#         )

#         self.video_transform = (
#             VideoTransform(
#                 subset="test",
#             )
#         )

#         print(
#             "Auto-AVSR Model Loaded.\n"
#         )

#     def predict(
#         self,
#         video_path: str,
#     ) -> str:

#         cap = cv2.VideoCapture(
#             video_path,
#         )

#         frames = []

#         while True:

#             ret, frame = cap.read()

#             if not ret:
#                 break

#             frame_rgb = (
#                 cv2.cvtColor(
#                     frame,
#                     cv2.COLOR_BGR2RGB,
#                 )
#             )

#             frames.append(
#                 frame_rgb,
#             )

#         cap.release()

#         if not frames:
#             return ""

#         video_np = np.array(
#             frames,
#         )

#         landmarks = (
#             self.landmarks_detector(
#                 video_np,
#             )
#         )

#         cropped = (
#             self.video_process(
#                 video_np,
#                 landmarks,
#             )
#         )

#         tensor_seq = (
#             torch.tensor(
#                 cropped,
#             )
#             .permute(
#                 (0, 3, 1, 2)
#             )
#         )

#         tensor_seq = (
#             self.video_transform(
#                 tensor_seq,
#             )
#         )

#         with torch.no_grad():

#             tensor_seq = (
#                 tensor_seq.to(
#                     self.device,
#                 )
#             )

#             prediction = (
#                 self.modelmodule(
#                     tensor_seq,
#                 )
#             )

#         return prediction.strip()


import os
import cv2
import torch
import numpy as np
import argparse

from lightning import ModelModule
from datamodule.transforms import VideoTransform
from preparation.detectors.mediapipe.detector import LandmarksDetector
from preparation.detectors.mediapipe.video_process import VideoProcess

class InferenceService:
    def __init__(self):
        print("\nLoading Auto-AVSR Model...")
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # FIX 1: Safely mock the expected argparse namespace
        parser = argparse.ArgumentParser()
        args, _ = parser.parse_known_args([])
        setattr(args, 'modality', 'video')

        self.modelmodule = ModelModule(args)

        # Make sure this path exactly matches where your .pth file is on your system
        checkpoint_path = "/home/anish/Desktop/Lipspeak/app/ml/auto_avsr/pretrained_models/vsr_trlrs2lrs3vox2avsp_base.pth"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model weights not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.modelmodule.model.load_state_dict(checkpoint)

        self.modelmodule.eval()
        self.modelmodule.to(self.device)

        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")

        print("Auto-AVSR Model Loaded.\n")

    def predict(self, video_path: str) -> str:
        # Prevent OpenCV from failing silently on bad paths
        if not os.path.exists(video_path):
            return "Error: Video file not found."

        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if not frames:
            return "Error: Unreadable or empty video."

        video_np = np.array(frames)

        # FIX 2: Graceful error handling for face tracking failures
        try:
            landmarks = self.landmarks_detector(video_np)
            cropped = self.video_process(video_np, landmarks)
        except Exception as e:
            print(f"Face tracking error: {e}")
            return "Error: Could not track face/lips in the provided video."

        # Process Tensors
        tensor_seq = torch.tensor(cropped).permute((0, 3, 1, 2))
        tensor_seq = self.video_transform(tensor_seq)

        with torch.no_grad():
            tensor_seq = tensor_seq.to(self.device)
            prediction = self.modelmodule(tensor_seq)

        return prediction.strip()