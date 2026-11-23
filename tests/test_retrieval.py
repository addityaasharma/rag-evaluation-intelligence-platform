from pathlib import Path

import pytest

from ragkit.ingestion import chunk_document, load_documents
from ragkit.retrieval import Retriever, reciprocal_rank_fusion

SAMPLE_DOCS = Path(__file__).parent.parent / "data" / "sample_docs"


def test_rrf_rewards_agreement(ranked):
    fused = reciprocal_rank_fusion([ranked("a", "b", "c"), ranked("b", "c", "d")], k=60)
    assert [r.chunk.id for r in fused][:2] == ["b", "c"]
    assert {r.chunk.id for r in fused} == {"a", "b", "c", "d"}


def test_rrf_empty():
    assert reciprocal_rank_fusion([[], []]) == []


@pytest.fixture
def indexed_store(store):
    chunks = [c for d in load_documents(SAMPLE_DOCS) for c in chunk_document(d, 800, 100)]
    store.add(chunks)
    return store


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_modes_find_relevant_source(indexed_store, mode):
    retriever = Retriever(indexed_store, mode=mode)
    results = retriever.retrieve("What is the API rate limit requests per minute?", k=3)
    assert results
    assert results[0].chunk.source == "rate_limits.md"


def test_empty_store_returns_nothing(store):
    for mode in ("dense", "bm25", "hybrid"):
        assert Retriever(store, mode=mode).retrieve("anything", k=3) == []


def test_bm25_index_refreshes_after_ingest(store, make_chunk):
    retriever = Retriever(store, mode="bm25")
    assert retriever.retrieve("zebra", k=1) == []
    store.add([make_chunk("1", "a zebra crossing")])
    assert retriever.retrieve("zebra", k=1)[0].chunk.id == "1"
