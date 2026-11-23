"""Retrieval metrics computed at chunk level against source-file relevance labels."""


def hit_rate(retrieved: list[str], relevant: set[str]) -> float:
    return float(any(source in relevant for source in retrieved))


def recall(retrieved: list[str], relevant: set[str]) -> float:
    return len(relevant & set(retrieved)) / len(relevant)


def precision(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved:
        return 0.0
    return sum(source in relevant for source in retrieved) / len(retrieved)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, source in enumerate(retrieved, start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def retrieval_scores(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    return {
        "hit_rate": hit_rate(retrieved, relevant),
        "recall": recall(retrieved, relevant),
        "precision": precision(retrieved, relevant),
        "mrr": reciprocal_rank(retrieved, relevant),
    }
