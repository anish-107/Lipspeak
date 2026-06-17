import sys
import os
import torch
import cv2
import numpy as np
import argparse

# Force python to see the root modules
sys.path.insert(0, "../")

from lightning import ModelModule
from datamodule.transforms import VideoTransform
from preparation.detectors.mediapipe.detector import LandmarksDetector
from preparation.detectors.mediapipe.video_process import VideoProcess

def run_inference(video_path, model_path):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return

    print("Step 1: Loading model to AWS GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args([])
    setattr(args, 'modality', 'video')
    
    # Initialize components
    modelmodule = ModelModule(args)
    ckpt = torch.load(model_path, map_location=device)
    modelmodule.model.load_state_dict(ckpt)
    modelmodule.eval()
    modelmodule.to(device)
    
    landmarks_detector = LandmarksDetector()
    video_process = VideoProcess(convert_gray=False)
    video_transform = VideoTransform(subset="test")
    
    print("Step 2: Loading video stream...")
    # Bypass torchvision entirely and use OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads in BGR, but PyTorch/Mediapipe expect RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    video_np = np.array(frames)
    print(f"Loaded video with {video_np.shape[0]} frames.")
    
    print("Step 3: Finding face landmarks and cropping mouth...")
    try:
        landmarks = landmarks_detector(video_np)
        cropped_mouth = video_process(video_np, landmarks)
    except Exception as e:
        print("❌ Face tracking failed! Make sure your face is visible from frame 1.")
        print(f"Details: {e}")
        return
        
    print("Step 4: Transforming tensors for deep learning execution...")
    # Convert back to torch and match dimensions [Frames, Channels, Height, Width]
    tensor_seq = torch.tensor(cropped_mouth).permute((0, 3, 1, 2))
    tensor_seq = video_transform(tensor_seq)
    
    print("Step 5: Evaluating lip-movement text transcription...")
    with torch.no_grad():
        tensor_seq = tensor_seq.to(device)
        prediction = modelmodule(tensor_seq)
        
    
    print("TRANSCRIBED SENTENCE:")
    print(f" {prediction.strip().upper()} ")
    

if __name__ == "__main__":
    # Ensure this points to where you downloaded/uploaded your video in Phase 1
    MY_VIDEO = "/home/anish/Downloads/4.mp4" 
    MODEL_WEIGHTS = "/home/anish/Desktop/Lipspeak/app/ml/auto_avsr/pretrained_models/vsr_trlrs2lrs3vox2avsp_base.pth"
    
    run_inference(MY_VIDEO, MODEL_WEIGHTS)
