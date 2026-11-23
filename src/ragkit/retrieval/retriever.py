import re

from rank_bm25 import BM25Okapi

from ragkit.config import RetrievalMode
from ragkit.models import Chunk, RetrievedChunk
from ragkit.retrieval.fusion import reciprocal_rank_fusion
from ragkit.vectorstore import ChromaStore

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Retriever:
    """Dense, lexical (BM25) or hybrid retrieval over a `ChromaStore`."""

    def __init__(
        self,
        store: ChromaStore,
        mode: RetrievalMode = "hybrid",
        candidate_pool: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self._store = store
        self._mode = mode
        self._pool = candidate_pool
        self._rrf_k = rrf_k
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[Chunk] = []
        self._bm25_tokens: list[set[str]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        pool = max(self._pool, k)
        if self._mode == "dense":
            return self._store.search(query, k)
        if self._mode == "bm25":
            return self._lexical(query, k)
        fused = reciprocal_rank_fusion(
            [self._store.search(query, pool), self._lexical(query, pool)], self._rrf_k
        )
        return fused[:k]

    def _lexical(self, query: str, limit: int) -> list[RetrievedChunk]:
        index = self._index()
        if index is None:
            return []
        query_tokens = _tokenize(query)
        query_tokens_set = set(query_tokens)
        scores = index.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        return [
            RetrievedChunk(chunk=self._bm25_chunks[i], score=float(scores[i]))
            for i in ranked
            if query_tokens_set & self._bm25_tokens[i]
        ]

    def _index(self) -> BM25Okapi | None:
        # Rebuilt only when the store size changes, e.g. after a new ingest.
        if len(self._bm25_chunks) != self._store.count():
            self._bm25_chunks = self._store.all_chunks()
            tokenized = [_tokenize(c.text) for c in self._bm25_chunks]
            self._bm25_tokens = [set(t) for t in tokenized]
            self._bm25 = BM25Okapi(tokenized) if tokenized else None
        return self._bm25
