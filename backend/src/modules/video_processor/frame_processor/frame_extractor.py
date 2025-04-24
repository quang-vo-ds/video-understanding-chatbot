import cv2
import os
from modules.utils import create_parent_folder, get_project_dir, extract_file_base_name

class FrameExtractor:
    @classmethod
    def invoke(self, video_path: str, frames_path: str = "data/frames", interval: int = 1):
        project_dir = get_project_dir()
        video_base_name = extract_file_base_name(video_path)
        abs_frames_path = os.path.join(project_dir, frames_path, video_base_name)
        create_parent_folder(abs_frames_path)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        saved = 0
        timestamps = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if int(count % round(fps * interval)) == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                filename = os.path.join(abs_frames_path, f"{saved:05d}.jpg")
                cv2.imwrite(filename, frame)
                timestamps.append((filename, timestamp))
                saved += 1
            count += 1
        cap.release()
        
        return timestamps