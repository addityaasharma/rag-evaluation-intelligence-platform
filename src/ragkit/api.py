from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from ragkit.ingestion.loaders import SUPPORTED_SUFFIXES, DocumentLoadError
from ragkit.models import ChatMessage, Verification
from ragkit.pipeline import RagPipeline, build_pipeline

_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=50)
    history: list[ChatMessage] = Field(default_factory=list)
    verify: bool = False


class SourceOut(BaseModel):
    source: str
    page: int | None
    score: float
    text: str
    cited: bool


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    verification: Verification | None = None


class DocumentOut(BaseModel):
    source: str
    chunks: int


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.pipeline = build_pipeline()
    yield


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(
        title="RAG Evaluation & Intelligence Platform", lifespan=None if pipeline else _lifespan
    )
    if pipeline:
        app.state.pipeline = pipeline

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Sync handlers: FastAPI runs them in a worker thread, so blocking LLM calls are fine.
    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> AskResponse:
        result = request.app.state.pipeline.ask(
            body.question, body.top_k, history=body.history, verify=body.verify
        )
        cited = set(result.cited_indices)
        return AskResponse(
            answer=result.text,
            sources=[
                SourceOut(
                    source=c.chunk.source,
                    page=c.chunk.page,
                    score=c.score,
                    text=c.chunk.text,
                    cited=i in cited,
                )
                for i, c in enumerate(result.contexts, start=1)
            ],
            verification=result.verification,
        )

    @app.get("/documents")
    def list_documents(request: Request) -> list[DocumentOut]:
        return [
            DocumentOut(source=d.source, chunks=d.chunks)
            for d in request.app.state.pipeline.documents()
        ]

    @app.post("/documents", status_code=201)
    def upload_document(file: UploadFile, request: Request) -> DocumentOut:
        name = file.filename or ""
        if not name.lower().endswith(tuple(SUPPORTED_SUFFIXES)):
            raise HTTPException(415, f"Supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
        data = file.file.read(_MAX_UPLOAD_BYTES + 1)
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File exceeds 25 MB limit")
        try:
            stats = request.app.state.pipeline.ingest_upload(name, data)
        except DocumentLoadError as exc:
            raise HTTPException(422, str(exc)) from exc
        if stats.documents == 0:
            raise HTTPException(422, "No extractable text found in file")
        return DocumentOut(source=name.rsplit("/", 1)[-1], chunks=stats.chunks)

    @app.delete("/documents", status_code=204)
    def delete_document(source: str, request: Request) -> None:
        request.app.state.pipeline.delete_document(source)

    return app
