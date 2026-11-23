from pydantic import BaseModel


class Document(BaseModel):
    id: str
    source: str
    text: str


class Chunk(BaseModel):
    id: str
    document_id: str
    source: str
    index: int
    text: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class Answer(BaseModel):
    question: str
    text: str
    contexts: list[RetrievedChunk]
