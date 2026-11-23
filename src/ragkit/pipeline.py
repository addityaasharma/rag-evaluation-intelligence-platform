from dataclasses import dataclass
from pathlib import Path

from ragkit.config import Settings, get_settings
from ragkit.embeddings import FastEmbedder
from ragkit.generation import generate_answer
from ragkit.ingestion import chunk_document, load_documents
from ragkit.llm import LLM, LiteLLMClient
from ragkit.models import Answer
from ragkit.retrieval import Retriever
from ragkit.vectorstore import ChromaStore


@dataclass(frozen=True)
class IngestStats:
    documents: int
    chunks: int


class RagPipeline:
    def __init__(self, settings: Settings, store: ChromaStore, retriever: Retriever, llm: LLM):
        self._settings = settings
        self._store = store
        self._retriever = retriever
        self._llm = llm

    def ingest(self, path: Path, *, reset: bool = False) -> IngestStats:
        if reset:
            self._store.reset()
        documents = load_documents(path)
        chunks = [
            chunk
            for doc in documents
            for chunk in chunk_document(
                doc, self._settings.chunk_size, self._settings.chunk_overlap
            )
        ]
        self._store.add(chunks)
        return IngestStats(documents=len(documents), chunks=len(chunks))

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        contexts = self._retriever.retrieve(question, top_k or self._settings.top_k)
        text = generate_answer(self._llm, question, contexts)
        return Answer(question=question, text=text, contexts=contexts)


def build_pipeline(settings: Settings | None = None) -> RagPipeline:
    settings = settings or get_settings()
    store = ChromaStore(
        FastEmbedder(settings.embedding_model),
        collection=settings.collection,
        path=settings.chroma_path,
    )
    retriever = Retriever(
        store,
        mode=settings.retrieval_mode,
        candidate_pool=settings.candidate_pool,
        rrf_k=settings.rrf_k,
    )
    return RagPipeline(settings, store, retriever, LiteLLMClient(settings.llm_model))
