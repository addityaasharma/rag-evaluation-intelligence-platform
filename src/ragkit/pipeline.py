import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ragkit.config import Settings, get_settings
from ragkit.embeddings import shared_embedder
from ragkit.generation import condense_question, generate_answer
from ragkit.ingestion import chunk_document, load_documents
from ragkit.llm import LLM, LiteLLMClient
from ragkit.models import Answer, ChatMessage, RetrievedChunk
from ragkit.retrieval import Retriever
from ragkit.vectorstore import ChromaStore
from ragkit.verification import HallucinationDetector, VerificationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestStats:
    documents: int
    chunks: int


@dataclass(frozen=True)
class IndexedDocument:
    source: str
    chunks: int


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        store: ChromaStore,
        retriever: Retriever,
        llm: LLM,
        verifier: HallucinationDetector | None = None,
    ):
        self._settings = settings
        self._store = store
        self._retriever = retriever
        self._llm = llm
        self._verifier = verifier

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

    def ingest_upload(self, filename: str, data: bytes) -> IngestStats:
        # Only the basename is used so a crafted filename cannot escape the temp dir.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / Path(filename).name
            path.write_bytes(data)
            return self.ingest(path)

    def documents(self) -> list[IndexedDocument]:
        counts: dict[str, int] = {}
        for chunk in self._store.all_chunks():
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        return [IndexedDocument(source, n) for source, n in sorted(counts.items())]

    def delete_document(self, source: str) -> None:
        self._store.delete_source(source)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self._retriever.retrieve(question, top_k or self._settings.top_k)

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        *,
        history: list[ChatMessage] | None = None,
        verify: bool = False,
    ) -> Answer:
        search_query = condense_question(self._llm, history, question) if history else question
        contexts = self.retrieve(search_query, top_k)
        text = generate_answer(self._llm, question, contexts, history)
        answer = Answer(question=question, text=text, contexts=contexts)
        if verify and self._verifier and contexts:
            try:
                answer.verification = self._verifier.check(text, contexts)
            except VerificationError:
                logger.warning("Hallucination check failed; returning answer unverified.")
        return answer


def build_pipeline(settings: Settings | None = None) -> RagPipeline:
    settings = settings or get_settings()
    store = ChromaStore(
        shared_embedder(settings.embedding_model),
        collection=settings.collection,
        path=settings.chroma_path,
    )
    retriever = Retriever(
        store,
        mode=settings.retrieval_mode,
        candidate_pool=settings.candidate_pool,
        rrf_k=settings.rrf_k,
    )
    verifier = HallucinationDetector(LiteLLMClient(settings.judge_model, max_tokens=1024))
    return RagPipeline(settings, store, retriever, LiteLLMClient(settings.llm_model), verifier)
