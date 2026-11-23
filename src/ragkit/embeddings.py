from functools import lru_cache
from typing import Any, Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Local ONNX embeddings via fastembed; no API key or GPU required."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._load().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(next(iter(self._load().query_embed(text))).tolist())

    def _load(self) -> Any:
        # Deferred so that importing the package does not download model weights.
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model


@lru_cache
def shared_embedder(model_name: str) -> FastEmbedder:
    """One loaded model per process, reused by every pipeline built from the same name."""
    return FastEmbedder(model_name)
