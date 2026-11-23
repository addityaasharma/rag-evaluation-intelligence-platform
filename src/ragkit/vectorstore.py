from itertools import batched
from pathlib import Path
from typing import Any

import chromadb

from ragkit.embeddings import Embedder
from ragkit.models import Chunk, RetrievedChunk

_BATCH_SIZE = 128


class ChromaStore:
    """Persistent cosine-similarity vector store backed by Chroma."""

    def __init__(
        self,
        embedder: Embedder,
        collection: str,
        path: Path | None = None,
        client: Any = None,
    ) -> None:
        self._embedder = embedder
        self._client: Any = client or chromadb.PersistentClient(path=str(path))
        self._collection: Any = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk]) -> None:
        for batch in batched(chunks, _BATCH_SIZE):
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self._embedder.embed_documents([c.text for c in batch]),
                metadatas=[
                    {"document_id": c.document_id, "source": c.source, "index": c.index}
                    for c in batch
                ],
            )

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=min(limit, total),
        )
        return [
            RetrievedChunk(chunk=_to_chunk(id_, text, meta), score=1.0 - distance)
            for id_, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def all_chunks(self) -> list[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            _to_chunk(id_, text, meta)
            for id_, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        ]

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )


def _to_chunk(id_: str, text: str, meta: Any) -> Chunk:
    return Chunk(
        id=id_,
        document_id=str(meta["document_id"]),
        source=str(meta["source"]),
        index=int(meta["index"]),
        text=text,
    )
