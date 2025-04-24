import os
import datetime

import numpy as np

from .frame_extractor import FrameExtractor
from .frame_captioner import FrameCaptioner
from modules.utils import get_project_dir, extract_file_base_name, create_parent_folder

class FrameProcessor:
    def __init__(self) -> None:
        self.project_dir = get_project_dir()
        self.frame_captioner = FrameCaptioner()
        self.frame_extractor = FrameExtractor

    def invoke(self, video_path: str, temp_data_dir: str = "data/frames") -> dict:
        frames_path = os.path.join(self.project_dir, temp_data_dir)
        create_parent_folder(frames_path)
        video_base_name = extract_file_base_name(video_path)

        frames = self.frame_extractor.invoke(
            video_path=video_path, 
            frames_path=frames_path
        )

        captions = self.frame_captioner.invoke(frames)

        output = [
            {
                "video_name": video_base_name,
                "text": c["caption"],
                "img_path": c["frame_path"],
                "start_time": str(datetime.timedelta(seconds=np.round(c["timestamp"]))),
                "end_time": str(datetime.timedelta(seconds=np.round(c["timestamp"])))
            } for c in captions]

        return output