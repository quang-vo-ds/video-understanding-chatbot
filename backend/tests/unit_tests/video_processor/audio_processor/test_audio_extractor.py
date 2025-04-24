import os
import numpy as np
import scipy
from pathlib import Path

from modules.video_processor.audio_processor.audio_extractor import AudioExtractor

PROJECT_DIR = Path(__file__).parents[5]

def test():
    video_path = "data/upload_video/Top 10 Beautiful Places to Visit in Sweden.mp4"
    audio_path = "data/upload_video/Top 10 Beautiful Places to Visit in Sweden.wav"

    AudioExtractor.invoke(
        video_path = os.path.join(PROJECT_DIR, video_path), 
        output_audio=os.path.join(PROJECT_DIR, audio_path)
    )

    audio_scipy = scipy.io.wavfile.read(os.path.join(PROJECT_DIR, audio_path))
    audio_arr = np.array(audio_scipy[1],dtype=float)

    print(audio_arr.dtype)
    print(audio_arr.shape)
    print(np.max(audio_arr))
    print(np.min(audio_arr))
    print(np.any(audio_arr))

    assert len(audio_arr) > 0
    assert np.any(audio_arr)