import logging
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from configs import settings
from modules.video_processor import AudioProcessor, FrameProcessor
from modules.embedder import TextEmbedder, ImageEmbedder
from modules.llm import LLMFactory
from modules.chat_engine import ChatEngine
from modules.vector_store import DBIngester, DBRetriever
from schemas import ProcessInput, QueryInput

app = FastAPI()

audio_processor = AudioProcessor()
frame_processor = FrameProcessor()

text_embedder = TextEmbedder()
img_embedder = ImageEmbedder()

ingester = DBIngester(
    text_embedder=text_embedder,
    img_embedder=img_embedder,
)
retriever = DBRetriever(
    text_embedder=text_embedder,
    img_embedder=img_embedder,
)

llm = LLMFactory.create(settings.LLM_PROVIDER)()
chat_engine = ChatEngine(llm=llm)

@app.post("/process")
async def rag(inp: ProcessInput):
    video_path = inp.video_path
    collection_name = inp.collection_name

    # Process Audio and Frame
    transcripts = audio_processor.invoke(video_path)
    captions = frame_processor.invoke(video_path)

    # Push data to vector store
    inp = {
        "collection_name": collection_name,
        "chunks": transcripts + captions,
    }

    output = ingester.insert(**inp)

    return JSONResponse(
        status_code=200,
        content={"insert_count": output["insert_count"]},
    )

@app.post("/chat")
async def chat(inp: QueryInput):
    collection_names = inp.collection_names
    text_query = inp.text_query
    image_query = inp.image_query

    context = retriever.query_embedding(
        collection_names=collection_names,
        text_query=text_query,
        image_query=image_query,
        search_top_k=10,
        sim_thres=0.3,
    )

    context_str = '\n'.join(json.dumps(c) for c in context)

    output = chat_engine.invoke(context=context_str, query=text_query)

    return JSONResponse(
        status_code=200,
        content={"generation": output},
    )
