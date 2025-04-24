import os
from pathlib import Path

from modules.video_processor.frame_processor.frame_extractor import FrameExtractor
from modules.video_processor.frame_processor.frame_captioner import FrameCaptioner

PROJECT_DIR = Path(__file__).parents[5]

video_path = "data/upload_video/Top 10 Beautiful Places to Visit in Sweden.mp4"
frames = FrameExtractor.invoke(video_path = os.path.join(PROJECT_DIR, video_path))

def test():
    captioner = FrameCaptioner()
    captions = captioner.invoke(frames)
    print(captions[:5])

    assert len(captions) > 0