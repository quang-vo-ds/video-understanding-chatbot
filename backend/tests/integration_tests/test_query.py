import os
import json
import requests
from pathlib import Path

URL = f"http://localhost:8080/chat"
COLLECTION_NAMES = ["test"]
TEXT_QUERY = "Why people should visit Stockholm?"
IMAGE_QUERY = None


def test():
    payload = {
        "collection_names": COLLECTION_NAMES,
        "text_query": TEXT_QUERY,
        "image_query": IMAGE_QUERY
    }

    res = requests.post(URL, json=payload)
    result = json.loads(res.content)

    print(result)

    assert len(result) > 0

if __name__ == '__main__':
    test()