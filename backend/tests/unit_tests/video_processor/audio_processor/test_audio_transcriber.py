import os
from pathlib import Path

from modules.video_processor.audio_processor.audio_extractor import AudioExtractor
from modules.video_processor.audio_processor.audio_transcriber import AudioTranscriber

PROJECT_DIR = Path(__file__).parents[5]

AUDIO_PATH = "data/upload_video/Top 10 Beautiful Places to Visit in Sweden - Sweden Travel Video.wav"

def test():
    transcriber = AudioTranscriber()
    audio_text = transcriber.invoke(os.path.join(PROJECT_DIR, AUDIO_PATH))
    print(audio_text[:10])

    assert len(audio_text) > 0

if __name__ == "__main__":
    test()