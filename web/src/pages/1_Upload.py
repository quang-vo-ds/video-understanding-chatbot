import os
import json
import requests
from pathlib import Path
import re

import streamlit as st
from streamlit.logger import get_logger

PROJECT_DIR = Path(__file__).parent
UPLOAD_DATA_DIR = "/data/upload_video"
Path(UPLOAD_DATA_DIR).mkdir(parents=True, exist_ok=True)

logger = get_logger(__name__)

PROCESS_URL = "http://backend:8080/process"

def send_request(url: str, payload: dict):
    res = requests.post(url, json=payload)
    result = json.loads(res.content)
    return result

def get_valid_milvus_name(name: str):
    return re.sub("[^0-9a-zA-Z]+", "_", name)

def main():
    title = "Upload Video"
    st.title(title)

    # Upload Video
    with st.form("Chatbot", clear_on_submit=True):
        file = st.file_uploader(
            "Please select a video to upload", 
            accept_multiple_files=False,
            type="mp4"
        )
        submitted = st.form_submit_button("UPLOAD")

        if submitted and file is not None:
            # Save file to /data/upload_video
            file_name = file.name
            binary_data = file.read()

            file_base_name = Path(file_name).stem
            collection_name = get_valid_milvus_name(file_base_name)

            local_file = os.path.join(UPLOAD_DATA_DIR, file_name)
            with open(local_file, "wb") as f:
                f.write(binary_data)

            # Process the uploaded file
            payload = {
                "collection_name": collection_name,
                "video_path": local_file
            }

            send_request(PROCESS_URL, payload)


if __name__ == "__main__":
    main()