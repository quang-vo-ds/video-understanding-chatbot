from modules.embedder import TextEmbedder, ImageEmbedder
from modules.vector_store import DBRetriever


COLLECTION_NAME = "Top_10_Beautiful_Places_to_Visit_in_Sweden"
DB_URL = "localhost:19530"
TEXT_QUERY = "Why people should visit Stockholm"
IMG_QUERY = "/home/quangvodc/video-chatbot/video-understanding-chatbot/data/frames/Top 10 Beautiful Places to Visit in Sweden/00001.jpg"

text_embedder = TextEmbedder()
img_embedder = ImageEmbedder()

def test():
    retriever = DBRetriever(
        text_embedder=text_embedder,
        img_embedder=img_embedder,
        db_url=DB_URL,
    )

    out = retriever.query_embedding(
        collection_names=[COLLECTION_NAME],
        text_query=TEXT_QUERY,
        image_query=IMG_QUERY,
        search_top_k=2,
        sim_thres=0.2,
    )

    print(len(out))
    print(out)

    assert len(out) > 0
