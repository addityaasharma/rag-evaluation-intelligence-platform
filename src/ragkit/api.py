from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from ragkit.pipeline import RagPipeline, build_pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=50)


class SourceOut(BaseModel):
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.pipeline = build_pipeline()
    yield


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(title="ragkit", lifespan=None if pipeline else _lifespan)
    if pipeline:
        app.state.pipeline = pipeline

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Sync handler: FastAPI runs it in a worker thread, so blocking LLM calls are fine.
    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> AskResponse:
        result = request.app.state.pipeline.ask(body.question, body.top_k)
        return AskResponse(
            answer=result.text,
            sources=[
                SourceOut(source=c.chunk.source, score=c.score, text=c.chunk.text)
                for c in result.contexts
            ],
        )

    return app
