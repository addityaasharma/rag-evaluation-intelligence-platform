from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["dense", "bm25", "hybrid"]


class Settings(BaseSettings):
    """Runtime configuration, overridable via `RAG_*` environment variables or `.env`."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Any LiteLLM model string. The default runs on Google AI Studio's free tier.
    llm_model: str = "gemini/gemini-flash-lite-latest"
    judge_model: str = "gemini/gemini-flash-lite-latest"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Free-tier Gemini allows 15 requests/minute per model; stay just under it.
    requests_per_minute: int = Field(12, gt=0)

    chroma_path: Path = Path(".rag/chroma")
    collection: str = "documents"
    history_path: Path = Path(".rag/history.db")

    chunk_size: int = Field(800, gt=0)
    chunk_overlap: int = Field(100, ge=0)

    top_k: int = Field(5, gt=0)
    candidate_pool: int = Field(20, gt=0)
    rrf_k: int = Field(60, gt=0)
    retrieval_mode: RetrievalMode = "hybrid"

    # Public hosting: isolate each browser session and cap usage of the shared API key.
    public_demo: bool = False
    demo_question_limit: int = Field(30, gt=0)

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    # Provider SDKs read API keys from the process environment, not from Settings.
    load_dotenv()
    return Settings()
