import os
import pytest
from pathlib import Path

from modules.embedder.image_embedder import ImageEmbedder

PROJECT_DIR = Path(__file__).parents[4]

def test():
    embedder = ImageEmbedder()
    image_path = "data/frames/Top 10 Beautiful Places to Visit in Sweden/00044.jpg"
    output = embedder.encode([None, os.path.join(PROJECT_DIR, image_path)])
    dense_vector = output[1]
    none_vector = output[0]
    
    print(embedder.dim)
    print(len(output))
    print(dense_vector[:5])
    print(len(dense_vector))
    print(none_vector[:5])
    print(len(none_vector))

    assert embedder.dim > 0
    assert len(dense_vector) > 0