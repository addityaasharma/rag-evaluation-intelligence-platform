from ragkit.models import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Merge ranked lists by summing 1 / (k + rank); scores in `rankings` are ignored.

    RRF needs no score normalisation, which makes it safe for combining cosine
    similarity with unbounded BM25 scores.
    """
    fused: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item.chunk.id] = fused.get(item.chunk.id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(item.chunk.id, item)

    ordered = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)
    return [RetrievedChunk(chunk=by_id[i].chunk, score=fused[i]) for i in ordered]
