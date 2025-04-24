import os
import numpy as np
import scipy
from pathlib import Path

from modules.video_processor.frame_processor import FrameProcessor

PROJECT_DIR = Path(__file__).parents[5]

def test():
    video_path = "data/upload_video/Top 10 Beautiful Places to Visit in Sweden.mp4"
    
    frame_processor = FrameProcessor()
    
    captions = frame_processor.invoke(os.path.join(PROJECT_DIR, video_path))

    print(captions[:2])
    print(captions[-2:])
    print(len(captions))

    assert len(captions) > 0