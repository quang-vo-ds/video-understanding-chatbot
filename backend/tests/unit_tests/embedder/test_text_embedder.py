import pytest

from modules.embedder.text_embedder import TextEmbedder

def test():
    embedder = TextEmbedder()
    query = "a river in the middle of a snowy forest"
    output = embedder.encode([query])
    vector = output[0]
    
    print(vector[:5])
    print(embedder.dim)
    print(len(vector))

    assert embedder.dim > 0
    assert len(vector) > 0
