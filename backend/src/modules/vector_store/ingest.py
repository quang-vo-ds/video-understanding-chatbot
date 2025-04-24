import logging
import json

from pymilvus import Collection, utility

from configs import settings
from modules.embedder import TextEmbedder, ImageEmbedder
from .setup import DBSetup


class DBIngester:
    """Class for building vector database"""

    def __init__(
        self,
        text_embedder: TextEmbedder,
        img_embedder: ImageEmbedder,
        db_url: str = settings.MILVUS_URL,
        logger: logging = logging.getLogger(settings.LOGGER),
    ) -> None:
        super().__init__()
        self.text_embedder = text_embedder
        self.img_embedder = img_embedder
        self.logger = logger

        DBSetup.connect_to_milvus(url=db_url)

    def insert(
        self,
        collection_name: str,
        chunks: list[dict],
        insert_batch: int = 1000,
    ) -> dict:
        if not chunks:
            return {"insert_count": 0}

        if not utility.has_collection(collection_name):
            DBSetup.create_collection(
                collection_name,
                text_embedder=self.text_embedder, 
                img_embedder=self.img_embedder
            )

        collection = Collection(collection_name)
        self.logger.info(f"Collection name: {collection_name}")

        text_embeddings = self.text_embedder.encode([c["text"] for c in chunks])
        img_embeddings = self.img_embedder.encode([c["img_path"] for c in chunks])

        data = [
            {
                "video_name": chunks[i]["video_name"],
                "content": json.dumps(chunks[i]),
                "text_vector": text_embeddings[i],
                "image_vector": img_embeddings[i],
            }
            for i in range(len(chunks))
        ]

        insert_count = 0
        for i in range(0, len(data), insert_batch):
            batch = data[i : i + insert_batch]
            res = collection.insert(data=batch)
            insert_count += res.insert_count

            self.logger.info(f"Insert result: {res}")

        return {"insert_count": insert_count}
