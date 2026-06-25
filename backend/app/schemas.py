from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    dataset: dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str
    rag_config: str = "hybrid_rerank"
    model: str = "openai_primary"
    alpha: float = Field(default=0.5, ge=0, le=1)
    retrieve_top_k: int = Field(default=20, ge=1, le=50)
    rerank_top_n: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[dict[str, str]]
    metrics: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    trace_id: str


class EvalRunRequest(BaseModel):
    run_name: str = "smoke-run"
    question_limit: int = Field(default=10, ge=1, le=100)
    question_ids: list[str] | None = None
    models: list[str] = Field(
        default_factory=lambda: [
            "groq_llama_3_1_8b",
            "groq_llama_3_3_70b",
            "groq_gpt_oss_20b",
            "openai_primary",
            "gemini_flash",
        ]
    )
    alphas: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75])
    retrieve_top_k: list[int] = Field(default_factory=lambda: [10])
    rerank_top_n: list[int] = Field(default_factory=lambda: [3, 5])


class EvalRunResponse(BaseModel):
    run_id: int
    status: str
    combinations: int
    questions: int
    total_results: int
    summary: dict[str, Any]
