import logging

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from configs import settings
from modules.embedder import TextEmbedder, ImageEmbedder


class DBSetup:
    """Class for init vector database and relevant collection"""

    @staticmethod
    def connect_to_milvus(url: str = settings.MILVUS_URL) -> None:
        hostname, port = url.split(":")
        try:
            connections.connect(
                host=hostname,
                port=port,
            )
        except Exception as e:
            raise

    @classmethod
    def create_collection(
        cls, collection_name: str, text_embedder: TextEmbedder, img_embedder: ImageEmbedder
    ) -> None:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()

            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="video_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(
                name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=text_embedder.dim
            ),
            FieldSchema(
                name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=img_embedder.dim
            )
        ]

        schema = CollectionSchema(fields)

        collection = Collection(collection_name, schema, consistency_level="Strong")

        cls._create_index(
            collection=collection,
            field_name="text_vector",
            index_type="AUTOINDEX",
            metric_type="IP",
        )

        cls._create_index(
            collection=collection,
            field_name="image_vector",
            index_type="AUTOINDEX",
            metric_type="IP",
        )

        collection.load()

        return

    @staticmethod
    def _create_index(
        collection: str, field_name: str, index_type: str, metric_type: str
    ) -> None:
        index = {"index_type": index_type, "metric_type": metric_type, "params": {}}
        collection.create_index(field_name, index)
        return
