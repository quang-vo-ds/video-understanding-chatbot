import os
from pathlib import Path

from modules.embedder import TextEmbedder, ImageEmbedder
from modules.vector_store import DBIngester

PROJECT_DIR = Path(__file__).parents[4]

CHUNKS = [
    {
        'video_name': 'Top 10 Beautiful Places to Visit in Sweden',
        'text': 'Sweden, the heart of Scandinavia, has an incredibly rich history and beautiful landscapes.', 
        'img_path': None, 
        'start_time': '0:00:16', 
        'end_time': '0:00:21'
    }, 
    {
        'video_name': 'Top 10 Beautiful Places to Visit in Sweden',
        'text': 'Sweden offers acres of unspoiled forests and majestic lakes to explore, not to mention vast archipelagos along its coasts.', 
        'img_path': None, 
        'start_time': '0:00:22', 
        'end_time': '0:00:30'
    },
    {
        'video_name': 'Top 10 Beautiful Places to Visit in Sweden',
        'text': 'a river in the middle of a snowy forest', 
        'img_path': os.path.join(PROJECT_DIR, "data/frames/Top 10 Beautiful Places to Visit in Sweden/00001.jpg"), 
        'start_time': '0:00:00', 
        'end_time': '0:00:01'
    },
    {
        'video_name': 'Top 10 Beautiful Places to Visit in Sweden',
        'text': 'the names of the famous movies', 
        'img_path': os.path.join(PROJECT_DIR, "data/frames/Top 10 Beautiful Places to Visit in Sweden/00550.jpg"), 
        'start_time': '0:09:09', 
        'end_time': '0:09:10'
    }, 
    {
        'video_name': 'Top 10 Beautiful Places to Visit in Sweden',
        'text': 'the names of the famous movies', 
        'img_path': os.path.join(PROJECT_DIR, "data/frames/Top 10 Beautiful Places to Visit in Sweden/00551.jpg"), 
        'start_time': '0:09:10', 
        'end_time': '0:09:11'
    }
    ]
COLLECTION_NAME = "Top_10_Beautiful_Places_to_Visit_in_Sweden"
DB_URL = "localhost:19530"

text_embedder = TextEmbedder()
img_embedder = ImageEmbedder()

def test():
    ingester = DBIngester(
        text_embedder=text_embedder,
        img_embedder=img_embedder,
        db_url=DB_URL,
    )

    inp = {
        "collection_name": COLLECTION_NAME,
        "chunks": CHUNKS,
    }

    output = ingester.insert(**inp)

    print(output)

    assert output["insert_count"] > 0
