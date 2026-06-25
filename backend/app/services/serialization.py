from __future__ import annotations

from app.core.types import AnswerResult, RetrievalCandidate


def serialize_candidate(candidate: RetrievalCandidate) -> dict:
    return {
        "chunk_id": candidate.chunk.chunk_id,
        "document": candidate.chunk.document,
        "section": candidate.chunk.section,
        "text": candidate.chunk.text,
        "dense_score": round(candidate.dense_score, 4),
        "sparse_score": round(candidate.sparse_score, 4),
        "hybrid_score": round(candidate.hybrid_score, 4),
        "rerank_score": round(candidate.rerank_score, 4),
        "metadata": candidate.chunk.metadata,
    }


def serialize_answer(answer: AnswerResult) -> dict:
    return {
        "answer": answer.answer,
        "citations": answer.citations,
        "model": answer.model,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "estimated_cost_usd": answer.estimated_cost_usd,
        "latency_ms": answer.latency_ms,
        "raw_response": answer.raw_response,
    }
