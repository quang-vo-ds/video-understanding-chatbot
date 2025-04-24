import numpy as np
from moviepy import VideoFileClip

class AudioExtractor:
    @classmethod
    def invoke(cls, video_path: str, output_audio: str) -> np.ndarray:
        clip = VideoFileClip(video_path)

        clip.audio.write_audiofile(output_audio)

        return output_audio
    
