import os
import json
import requests
from pathlib import Path

URL = f"http://localhost:8080/process"
VIDEO_PATH = "/data/upload_video/Top 10 Beautiful Places to Visit in Sweden.mp4"
COLLECTION_NAME = "test"


def test():
    payload = {
        "collection_name": COLLECTION_NAME,
        "video_path": VIDEO_PATH
    }

    res = requests.post(URL, json=payload)
    result = json.loads(res.content)

    print(result)

    assert len(result) > 0

if __name__ == '__main__':
    test()