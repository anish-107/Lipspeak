
import sys
sys.path.insert(0, "../")

import os
import torch
import torchaudio
import torchvision

import os
from lightning import ModelModule
from datamodule.transforms import AudioTransform, VideoTransform

import argparse
parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args(args=[])

class InferencePipeline(torch.nn.Module):
    def __init__(self, args, ckpt_path, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        self.modality = args.modality
        if self.modality == "audio":
            self.audio_transform = AudioTransform(subset="test")
        elif self.modality == "video":
            if detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            elif detector == "retinaface":
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector(device="cuda:0")
                self.video_process = VideoProcess(convert_gray=False)
            self.video_transform = VideoTransform(subset="test")

        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.modelmodule = ModelModule(args)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.eval()

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio)
            with torch.no_grad():
                transcript = self.modelmodule(audio)

        if self.modality == "video":
            video = self.load_video(data_filename)
            landmarks = self.landmarks_detector(video)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            video = self.video_transform(video)
            with torch.no_grad():
                transcript = self.modelmodule(video)

        return transcript

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

!wget --content-disposition http://www.doc.ic.ac.uk/~pm4115/autoAVSR/autoavsr_demo_video.mp4 -O ./input.mp4
data_filename = "input.mp4"

!wget http://www.doc.ic.ac.uk/~pm4115/autoAVSR/vsr_trlrs3_base.pth -O ./vsr_trlrs3_base.pth
model_path = "./vsr_trlrs3_base.pth"

setattr(args, 'modality', 'video')
pipeline = InferencePipeline(args, model_path, detector="retinaface")

transcript = pipeline("input.mp4")
print(transcript)

!wget http://www.doc.ic.ac.uk/~pm4115/autoAVSR/asr_trlrs3_base.pth -O ./asr_trlrs3_base.pth
model_path = "./asr_trlrs3_base.pth"

setattr(args, 'modality', 'audio')
pipeline = InferencePipeline(args, model_path)

transcript = pipeline("input.mp4")
print(transcript)
