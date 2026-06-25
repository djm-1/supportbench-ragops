from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SupportChunk:
    chunk_id: str
    document: str
    section: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalCandidate:
    chunk: SupportChunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class EvalQuestion:
    id: str
    question: str
    reference_answer: str
    expected_doc: str | None
    expected_section: str | None
    tags: list[str]
    should_refuse: bool = False
    question_type: str = "direct"
    expected_sources: list[dict[str, str]] = field(default_factory=list)
    reference_facts: list[str] = field(default_factory=list)
    evaluation_notes: str = ""


@dataclass
class RAGConfig:
    model: str
    alpha: float
    retrieve_top_k: int
    rerank_top_n: int


@dataclass
class AnswerResult:
    answer: str
    citations: list[dict[str, str]]
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: int
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerJudgeResult:
    score: float
    label: str
    rationale: str
    missing_facts: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    judge_model: str = "deterministic_fallback"
    judge_prompt_version: str = "answer-judge-v1"
    judge_latency_ms: int = 0
    judge_cost_usd: float = 0.0
    judge_unavailable: bool = False
    judge_source: str = "deterministic"


@dataclass
class EvalMetrics:
    answer_correctness: float
    groundedness: float
    citation_accuracy: float
    refusal_correctness: float
    retrieval_hit: float
    mrr: float
    quality_score: float
    failure_category: str
    result_label: str = "wrong"
    issue_label: str = "unsupported_answer"
    judge_label: str = "not_run"
    judge_rationale: str = ""
    missing_facts: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    judge_model: str = "deterministic_fallback"
    judge_prompt_version: str = "answer-judge-v1"
    judge_latency_ms: int = 0
    judge_cost_usd: float = 0.0
    judge_unavailable: bool = False
    source_recall: float = 0.0
    citation_matched: bool = False
    retrieved_expected_sources: list[dict[str, str]] = field(default_factory=list)
    required_expected_sources: list[dict[str, str]] = field(default_factory=list)
    judge_source: str = "deterministic"
    deterministic_gate_failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer_correctness": self.answer_correctness,
            "groundedness": self.groundedness,
            "citation_accuracy": self.citation_accuracy,
            "refusal_correctness": self.refusal_correctness,
            "retrieval_hit": self.retrieval_hit,
            "mrr": self.mrr,
            "quality_score": self.quality_score,
            "failure_category": self.failure_category,
            "result_label": self.result_label,
            "issue_label": self.issue_label,
            "judge_label": self.judge_label,
            "judge_rationale": self.judge_rationale,
            "missing_facts": self.missing_facts,
            "contradictions": self.contradictions,
            "judge_model": self.judge_model,
            "judge_prompt_version": self.judge_prompt_version,
            "judge_latency_ms": self.judge_latency_ms,
            "judge_cost_usd": self.judge_cost_usd,
            "judge_unavailable": self.judge_unavailable,
            "source_recall": self.source_recall,
            "citation_matched": self.citation_matched,
            "retrieved_expected_sources": self.retrieved_expected_sources,
            "required_expected_sources": self.required_expected_sources,
            "judge_source": self.judge_source,
            "deterministic_gate_failures": self.deterministic_gate_failures,
        }
