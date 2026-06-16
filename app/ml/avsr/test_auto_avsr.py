import argparse

from inference_pipeline import (
    InferencePipeline,
)

args = argparse.Namespace()
args.modality = "video"

pipeline = InferencePipeline(
    args,
    "checkpoints/vsr_trlrs3_base.pth",
    detector="mediapipe",
)

print(
    pipeline(
        "bbbs4n.mpg"
    )
)