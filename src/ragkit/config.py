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

    # Any LiteLLM model string, e.g. "anthropic/claude-sonnet-5-5" or "openai/gpt-4o".
    llm_model: str = "anthropic/claude-sonnet-5-5"
    judge_model: str = "anthropic/claude-sonnet-5-5"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chroma_path: Path = Path(".rag/chroma")
    collection: str = "documents"
    history_path: Path = Path(".rag/history.db")

    chunk_size: int = Field(800, gt=0)
    chunk_overlap: int = Field(100, ge=0)

    top_k: int = Field(5, gt=0)
    candidate_pool: int = Field(20, gt=0)
    rrf_k: int = Field(60, gt=0)
    retrieval_mode: RetrievalMode = "hybrid"

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
