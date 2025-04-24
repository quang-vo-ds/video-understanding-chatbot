import logging
import time

from pymilvus import AnnSearchRequest, Collection, WeightedRanker, utility
from pymilvus.client.abstract import SearchResult

from configs import settings
from modules.embedder import TextEmbedder, ImageEmbedder

from .setup import DBSetup


class DBRetriever:
    """Class for retrieving data from vector database"""

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

    def query_embedding(
        self,
        collection_names: list[str],
        text_query: str | None = None,
        image_query: str | None = None,
        search_top_k: int = 5,
        sim_thres: float = 0.6,
    ) -> list[dict]:
        """Query based on embedding similarity"""

        text_chunks = []

        # Perform hybrid search for each collection
        for collection_name in collection_names:
            if not utility.has_collection(collection_name):
                self.logger.info(f"{collection_name} does not exist")
                continue

            collection = Collection(collection_name)

            start = time.time()
            text_embedding = self.text_embedder.encode([text_query])[0]
            img_embedding = self.img_embedder.encode([image_query])[0]

            search_results = self._hybrid_search(
                col=collection,
                text_embedding=text_embedding,
                image_embedding=img_embedding,
                text_weight=1.0,
                image_weight=0.5,
                search_top_k=search_top_k,
                sim_thres=sim_thres,
            )
            end = time.time()
            self.logger.info(f"Search took: {end-start}")

            text_chunks.extend(search_results)


        # Log chunk info
        if text_chunks:
            self.logger.info(f"Number of chunks: {len(text_chunks)}")
            self.logger.info(f"Chunks 0: {text_chunks[0]}")

        return text_chunks

    def _hybrid_search(
        self,
        col: Collection,
        text_embedding: list[float],
        image_embedding: list[float],
        text_weight: float,
        image_weight: float,
        search_top_k: int,
        sim_thres: float,
    ):
        text_req = AnnSearchRequest(
            [text_embedding],
            "text_vector",
            {"metric_type": "IP", "params": {}},
            limit=search_top_k,
        )

        img_req = AnnSearchRequest(
            [image_embedding],
            "image_vector",
            {"metric_type": "IP", "params": {}},
            limit=search_top_k,
        )

        rerank = WeightedRanker(text_weight, image_weight)

        res = col.hybrid_search(
            [text_req, img_req],
            rerank=rerank,
            limit=search_top_k,
            output_fields=["video_name", "content"],
        )

        return self._extract_milvus_result(res, sim_thres=sim_thres)
    
    
    @staticmethod
    def _extract_milvus_result(
        search_res: SearchResult, sim_thres: float
    ) -> list[dict]:
        out = [
            {
                "video_name": hit.entity.get("video_name"),
                "content": hit.entity.get("content"),
                "sim_score": distance
            }
            for hits in search_res
            for hit in hits
            if (distance := hit.distance) >= sim_thres
        ]
        return out
