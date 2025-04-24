import os

from .audio_extractor import AudioExtractor
from .audio_transcriber import AudioTranscriber
from modules.utils import get_project_dir, extract_file_base_name, create_parent_folder

class AudioProcessor:
    def __init__(self) -> None:
        self.project_dir = get_project_dir()
        self.audio_transcriber = AudioTranscriber()
        self.audio_extractor = AudioExtractor

    def invoke(self, video_path: str, temp_data_dir: str = "data/audio") -> dict:
        video_base_name = extract_file_base_name(video_path)
        audio_path = os.path.join(self.project_dir, temp_data_dir, f"{video_base_name}.mp3")
        create_parent_folder(audio_path)

        self.audio_extractor.invoke(
            video_path = video_path, 
            output_audio=audio_path
        )

        transcripts = self.audio_transcriber.invoke(audio_path)

        output = [
            {
                "video_name": video_base_name,
                "text": t["sentence"],
                "img_path": None,
                "start_time": t["start_time"],
                "end_time": t["end_time"]
            } for t in transcripts]

        return output