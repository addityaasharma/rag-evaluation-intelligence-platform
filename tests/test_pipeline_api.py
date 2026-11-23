from pathlib import Path

from conftest import FakeLLM, make_pdf
from fastapi.testclient import TestClient

from ragkit.api import create_app
from ragkit.config import Settings
from ragkit.models import ChatMessage
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


def test_followup_question_is_condensed_before_retrieval(store):
    def reply(system: str, user: str) -> str:
        return "How long are backups retained?" if system.startswith("Rewrite") else "35 days [1]"

    llm = FakeLLM(reply)
    pipeline = make_pipeline(store, llm)
    pipeline.ingest(DOCS)
    history = [
        ChatMessage(role="user", content="How are backups handled?"),
        ChatMessage(role="assistant", content="They are kept."),
    ]

    answer = pipeline.ask("And for how long?", history=history)

    assert answer.contexts[0].chunk.source == "security.md"
    assert "Conversation so far" in llm.calls[-1][1]


def test_verify_attaches_verification_and_tolerates_failure(store):
    from ragkit.verification import HallucinationDetector

    ok = '{"claims": [{"text": "x", "supported": false}]}'
    pipeline = make_pipeline(store, FakeLLM())
    pipeline._verifier = HallucinationDetector(FakeLLM(ok))
    pipeline.ingest(DOCS)
    assert pipeline.ask("backups", verify=True).verification.has_hallucination
    assert pipeline.ask("backups").verification is None

    pipeline._verifier = HallucinationDetector(FakeLLM("garbage"))
    assert pipeline.ask("backups", verify=True).verification is None


def test_upload_list_and_delete_documents(store):
    pipeline = make_pipeline(store, FakeLLM())
    pipeline.ingest_upload("../../evil/notes.txt", b"The vault code is 1234.")
    assert [(d.source, d.chunks) for d in pipeline.documents()] == [("notes.txt", 1)]
    assert pipeline.retrieve("vault code", 1)[0].chunk.source == "notes.txt"

    pipeline.delete_document("notes.txt")
    assert pipeline.documents() == []
    assert pipeline.retrieve("vault code", 1) == []


def test_api_document_endpoints(store):
    client = TestClient(create_app(make_pipeline(store, FakeLLM())))

    created = client.post(
        "/documents", files={"file": ("a.pdf", make_pdf(["Quarterly revenue grew"]))}
    )
    assert created.status_code == 201
    assert client.get("/documents").json() == [{"source": "a.pdf", "chunks": 1}]

    body = client.post("/ask", json={"question": "revenue growth", "verify": False}).json()
    assert body["sources"][0]["page"] == 1
    assert body["sources"][0]["cited"] is True

    assert client.post("/documents", files={"file": ("x.exe", b"MZ")}).status_code == 415
    assert client.post("/documents", files={"file": ("bad.pdf", b"nope")}).status_code == 422
    assert client.delete("/documents", params={"source": "a.pdf"}).status_code == 204
    assert client.get("/documents").json() == []


def test_rate_limiter_spaces_calls():
    import time

    from ragkit.llm import RateLimiter

    limiter = RateLimiter(requests_per_minute=1200)  # 50 ms apart
    start = time.monotonic()
    for _ in range(4):
        limiter.wait()
    assert time.monotonic() - start >= 0.14
