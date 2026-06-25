from __future__ import annotations

from app.core.text import token_set
from app.core.types import RetrievalCandidate


class LexicalReranker:
    """Fast deterministic reranker for local demos.

    The interface mirrors a cross-encoder reranker, but the implementation uses
    query/chunk token overlap so the app can run without model downloads.
    """

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        *,
        limit: int,
    ) -> list[RetrievalCandidate]:
        query_tokens = token_set(query)
        for candidate in candidates:
            chunk_tokens = token_set(candidate.chunk.text + " " + candidate.chunk.section)
            overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            candidate.rerank_score = 0.7 * overlap + 0.3 * candidate.hybrid_score
        return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)[:limit]
