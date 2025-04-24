import os
import numpy as np
import scipy
from pathlib import Path

from modules.video_processor.audio_processor import AudioProcessor

PROJECT_DIR = Path(__file__).parents[5]

def test():
    video_path = "data/upload_video/Top 10 Beautiful Places to Visit in Sweden.mp4"
    
    audio_processor = AudioProcessor()
    
    transcripts = audio_processor.invoke(os.path.join(PROJECT_DIR, video_path))

    print(transcripts[:2])
    print(transcripts[-2:])

    assert len(transcripts) > 0