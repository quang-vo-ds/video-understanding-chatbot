import os
from pathlib import Path

from modules.video_processor.frame_processor.frame_extractor import FrameExtractor

PROJECT_DIR = Path(__file__).parents[5]

def test():
    video_path = "data/upload_video/Top 10 Beautiful Places to Visit in Sweden.mp4"
    frames = FrameExtractor.invoke(video_path = os.path.join(PROJECT_DIR, video_path))
    print(frames[:5])

    assert len(frames) > 0