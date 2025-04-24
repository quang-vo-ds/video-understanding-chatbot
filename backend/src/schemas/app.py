from pydantic import BaseModel


class ProcessInput(BaseModel):
    collection_name: str
    video_path: str | None = None

class QueryInput(BaseModel):
    collection_names: list[str]
    text_query: str | None = None
    image_query: str | None = None