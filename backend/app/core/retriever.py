from __future__ import annotations

from app.core.bm25 import BM25Index
from app.core.dense_index import build_dense_index
from app.core.fusion import fuse_candidates
from app.core.reranker import LexicalReranker
from app.core.types import RetrievalCandidate, SupportChunk


class HybridRetriever:
    def __init__(self, chunks: list[SupportChunk]) -> None:
        self.chunks = chunks
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.dense_index = build_dense_index(chunks)
        self.bm25 = BM25Index([chunk.text + " " + chunk.section for chunk in chunks])
        self.reranker = LexicalReranker()

    def retrieve(
        self,
        query: str,
        *,
        alpha: float,
        retrieve_top_k: int,
        rerank_top_n: int,
    ) -> list[RetrievalCandidate]:
        dense_scores = self.dense_index.search(query, limit=max(retrieve_top_k * 2, retrieve_top_k))
        sparse_scores = self.bm25.search(query, limit=max(retrieve_top_k * 2, retrieve_top_k))

        candidate_map: dict[str, RetrievalCandidate] = {}
        for chunk_id, score in dense_scores:
            chunk = self.chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            candidate_map[chunk_id] = RetrievalCandidate(chunk=chunk, dense_score=score)
        for index, score in sparse_scores:
            chunk = self.chunks[index]
            candidate = candidate_map.setdefault(chunk.chunk_id, RetrievalCandidate(chunk=chunk))
            candidate.sparse_score = score

        fused = fuse_candidates(list(candidate_map.values()), alpha=alpha)[:retrieve_top_k]
        return self.reranker.rerank(query, fused, limit=rerank_top_n)
