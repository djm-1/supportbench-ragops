from __future__ import annotations

from app.core.types import RetrievalCandidate


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(value - low) / (high - low) for value in values]


def fuse_candidates(candidates: list[RetrievalCandidate], alpha: float) -> list[RetrievalCandidate]:
    alpha = max(0.0, min(1.0, alpha))
    dense_scores = _normalize([candidate.dense_score for candidate in candidates])
    sparse_scores = _normalize([candidate.sparse_score for candidate in candidates])
    for candidate, dense_score, sparse_score in zip(candidates, dense_scores, sparse_scores):
        candidate.hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score
    return sorted(candidates, key=lambda item: item.hybrid_score, reverse=True)
