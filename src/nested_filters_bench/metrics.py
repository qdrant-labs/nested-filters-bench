import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Quality:
    """Search quality @k, averaged over queries. All values are between 0 and 1."""

    recall: float
    mrr: float
    ndcg: float


def recall_at_k(retrieved: list[int], truth: list[int], k: int) -> float:
    """Share of the exact top-k that the search returned."""
    relevant = set(truth[:k])
    return len(relevant & set(retrieved[:k])) / len(relevant)


def mrr_at_k(retrieved: list[int], truth: list[int], k: int) -> float:
    """1 / rank of the first returned point that is in the exact top-k (0 if none)."""
    relevant = set(truth[:k])
    for rank, point_id in enumerate(retrieved[:k], start=1):
        if point_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], truth: list[int], k: int) -> float:
    """nDCG with binary relevance: a hit is any point in the exact top-k."""
    relevant = set(truth[:k])
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, point_id in enumerate(retrieved[:k], start=1)
        if point_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, len(relevant) + 1))
    return dcg / ideal


def mean_quality(retrieved: list[list[int]], truth: list[list[int]], k: int) -> Quality:
    """Average the three metrics over all queries."""
    n = len(retrieved)
    pairs = list(zip(retrieved, truth, strict=True))
    return Quality(
        recall=sum(recall_at_k(r, t, k) for r, t in pairs) / n,
        mrr=sum(mrr_at_k(r, t, k) for r, t in pairs) / n,
        ndcg=sum(ndcg_at_k(r, t, k) for r, t in pairs) / n,
    )
