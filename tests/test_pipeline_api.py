from pathlib import Path

from conftest import FakeLLM
from fastapi.testclient import TestClient

from ragkit.api import create_app
from ragkit.config import Settings
from ragkit.pipeline import RagPipeline
from ragkit.retrieval import Retriever

DOCS = Path(__file__).parent.parent / "data" / "sample_docs"


def make_pipeline(store, llm):
    return RagPipeline(Settings(), store, Retriever(store), llm)


def test_ingest_reports_stats_and_is_idempotent(store):
    pipeline = make_pipeline(store, FakeLLM())
    first = pipeline.ingest(DOCS)
    pipeline.ingest(DOCS)
    assert first.documents == 4
    assert store.count() == first.chunks


def test_ask_puts_retrieved_context_in_prompt(store):
    llm = FakeLLM()
    pipeline = make_pipeline(store, llm)
    pipeline.ingest(DOCS)

    answer = pipeline.ask("How long are backups retained?")

    assert answer.text == "stub answer [1]"
    assert answer.contexts[0].chunk.source == "security.md"
    assert "35 days" in llm.calls[0][1]


def test_api_ask_and_health(store):
    pipeline = make_pipeline(store, FakeLLM())
    pipeline.ingest(DOCS)
    client = TestClient(create_app(pipeline))

    assert client.get("/health").json() == {"status": "ok"}
    body = client.post("/ask", json={"question": "Which plans support SAML?", "top_k": 2}).json()
    assert body["answer"] == "stub answer [1]"
    assert len(body["sources"]) == 2


def test_api_rejects_empty_question(store):
    client = TestClient(create_app(make_pipeline(store, FakeLLM())))
    assert client.post("/ask", json={"question": ""}).status_code == 422
