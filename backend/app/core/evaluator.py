from __future__ import annotations

import re

from app.core.judge import deterministic_judge
from app.core.text import clamp, overlap_ratio, token_set
from app.core.types import AnswerJudgeResult, AnswerResult, EvalMetrics, EvalQuestion, RetrievalCandidate

REFUSAL_MARKERS = (
    "do not state",
    "do not specify",
    "do not mention",
    "not enough",
    "provided docs do not",
    "provided support docs do not",
    "cannot determine",
    "i don't know",
)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


def _expected_sources(question: EvalQuestion) -> list[dict[str, str]]:
    if question.should_refuse:
        return []
    sources = [
        {"document": str(source.get("document")), "section": str(source.get("section"))}
        for source in question.expected_sources
        if source.get("document") and source.get("section")
    ]
    if sources:
        return sources
    if question.expected_doc and question.expected_section:
        return [{"document": question.expected_doc, "section": question.expected_section}]
    return []


def _source_key(source: dict[str, str]) -> tuple[str, str]:
    return source["document"], source["section"]


def _citation_accuracy(question: EvalQuestion, answer: AnswerResult) -> tuple[float, bool]:
    if question.should_refuse:
        score = 1.0 if not answer.citations else 0.0
        return score, score == 1.0
    required_sources = _expected_sources(question)
    if not required_sources:
        return 1.0, True
    if not answer.citations:
        return 0.0, False
    cited = {
        (str(citation.get("document")), str(citation.get("section")))
        for citation in answer.citations
    }
    matched = sum(1 for source in required_sources if _source_key(source) in cited)
    score = matched / max(1, len(required_sources))
    return score, score == 1.0


def _retrieval_scores(
    question: EvalQuestion,
    candidates: list[RetrievalCandidate],
) -> tuple[float, float, float, list[dict[str, str]], list[dict[str, str]]]:
    required_sources = _expected_sources(question)
    if question.should_refuse or not required_sources:
        return 1.0, 1.0, 1.0, [], required_sources
    reciprocal_ranks: list[float] = []
    retrieved_sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    required_keys = {_source_key(source) for source in required_sources}
    for index, candidate in enumerate(candidates, start=1):
        key = (candidate.chunk.document, candidate.chunk.section)
        if key in required_keys and key not in seen:
            seen.add(key)
            retrieved_sources.append({"document": key[0], "section": key[1]})
            reciprocal_ranks.append(1 / index)
    source_recall = len(seen) / max(1, len(required_sources))
    retrieval_hit = 1.0 if source_recall == 1.0 else 0.0
    mrr = sum(reciprocal_ranks) / max(1, len(required_sources))
    return retrieval_hit, mrr, source_recall, retrieved_sources, required_sources


def _citation_failure_category(question: EvalQuestion, answer: AnswerResult, citation_accuracy: float) -> str:
    if question.should_refuse and answer.citations:
        return "wrong_citation"
    if not answer.citations:
        return "missing_citation"
    if citation_accuracy == 0:
        return "wrong_citation"
    return "partial_citation"


def _gate_failures(
    *,
    retrieval_hit: float,
    citation_accuracy: float,
    refusal_correctness: float,
    answer_correctness: float,
    groundedness: float,
    failure_category: str,
    grounding_required: bool = True,
) -> list[str]:
    failures: list[str] = []
    if retrieval_hit < 1:
        failures.append("retrieval")
    if citation_accuracy < 1:
        failures.append("citation")
    if refusal_correctness < 1:
        failures.append("refusal")
    if answer_correctness < 0.65:
        failures.append("answer")
    if grounding_required and groundedness < 0.55:
        failures.append("grounding")
    if not failures and failure_category != "passed":
        failures.append(failure_category)
    return failures


def _groundedness(answer: AnswerResult, candidates: list[RetrievalCandidate]) -> float:
    context = " ".join(candidate.chunk.text for candidate in candidates)
    answer_tokens = token_set(answer.answer)
    context_tokens = token_set(context)
    if not answer_tokens:
        return 0.0
    return clamp(len(answer_tokens & context_tokens) / max(1, len(answer_tokens)))


def _result_issue(
    failure_category: str,
    *,
    question: EvalQuestion,
    judge_result: AnswerJudgeResult,
) -> tuple[str, str]:
    if failure_category == "passed":
        return "right", "none"
    if failure_category == "bad_retrieval":
        return "wrong", "source_not_found"
    if failure_category in {"missing_citation", "partial_citation"}:
        return "partial", "citation_issue"
    if failure_category in {"wrong_citation", "unsupported_citation"}:
        return "wrong", "citation_issue"
    if failure_category == "partial_answer":
        return "partial", "incomplete_answer"
    if failure_category == "needs_review":
        return "partial", "provider_error" if judge_result.judge_unavailable else "incomplete_answer"
    if failure_category in {"should_have_refused", "ungrounded_answer"}:
        return "wrong", "unsupported_answer"
    if failure_category == "incorrect_refusal":
        return "wrong", "incomplete_answer"
    if question.should_refuse and failure_category == "wrong_answer":
        return "wrong", "unsupported_answer"
    return "wrong", "unsupported_answer"


def evaluate_answer(
    question: EvalQuestion,
    answer: AnswerResult,
    candidates: list[RetrievalCandidate],
    judge_result: AnswerJudgeResult | None = None,
) -> EvalMetrics:
    answer_lower = answer.answer.lower()
    refused = any(marker in answer_lower for marker in REFUSAL_MARKERS)
    refusal_correctness = 1.0 if refused == question.should_refuse else 0.0

    judge_result = judge_result or deterministic_judge(question, answer)
    answer_correctness = judge_result.score
    if not question.should_refuse and judge_result.judge_model == "deterministic_fallback":
        lexical_overlap = overlap_ratio(answer.answer, question.reference_answer)
        reference_numbers = _numbers(question.reference_answer)
        if reference_numbers:
            number_score = len(reference_numbers & _numbers(answer.answer)) / len(reference_numbers)
            answer_correctness = max(answer_correctness, clamp(0.65 * lexical_overlap + 0.35 * number_score))

    citation_accuracy, citation_matched = _citation_accuracy(question, answer)
    retrieval_hit, mrr, source_recall, retrieved_expected_sources, required_expected_sources = _retrieval_scores(question, candidates)
    groundedness = _groundedness(answer, candidates)
    grounding_required = not question.should_refuse
    if question.should_refuse and refusal_correctness == 1.0:
        groundedness = 1.0
    if (
        not question.should_refuse
        and retrieval_hit == 1.0
        and citation_accuracy == 1.0
        and judge_result.label in {"correct", "partial"}
    ):
        groundedness = max(groundedness, 0.78 if judge_result.label == "correct" else 0.58)
    quality_score = (
        0.40 * answer_correctness
        + 0.25 * groundedness
        + 0.20 * citation_accuracy
        + 0.15 * refusal_correctness
    )

    if retrieval_hit == 0:
        failure_category = "bad_retrieval"
    elif refusal_correctness == 0:
        failure_category = "should_have_refused" if question.should_refuse else "incorrect_refusal"
    elif citation_accuracy == 0:
        failure_category = _citation_failure_category(question, answer, citation_accuracy)
    elif citation_accuracy < 1:
        failure_category = "partial_citation"
    elif judge_result.judge_unavailable:
        failure_category = "needs_review"
    elif answer_correctness < 0.65:
        failure_category = "wrong_answer"
    elif judge_result.label == "partial":
        failure_category = "partial_answer"
    elif (
        judge_result.judge_source != "llm"
        and judge_result.judge_model == "deterministic_fallback"
        and judge_result.label != "correct"
    ):
        failure_category = "needs_review"
    elif grounding_required and groundedness < 0.55:
        failure_category = "ungrounded_answer"
    else:
        failure_category = "passed"

    result_label, issue_label = _result_issue(
        failure_category,
        question=question,
        judge_result=judge_result,
    )
    deterministic_gate_failures = _gate_failures(
        retrieval_hit=retrieval_hit,
        citation_accuracy=citation_accuracy,
        refusal_correctness=refusal_correctness,
        answer_correctness=answer_correctness,
        groundedness=groundedness,
        failure_category=failure_category,
        grounding_required=grounding_required,
    )

    return EvalMetrics(
        answer_correctness=round(answer_correctness, 4),
        groundedness=round(groundedness, 4),
        citation_accuracy=round(citation_accuracy, 4),
        refusal_correctness=round(refusal_correctness, 4),
        retrieval_hit=round(retrieval_hit, 4),
        mrr=round(mrr, 4),
        quality_score=round(quality_score, 4),
        failure_category=failure_category,
        result_label=result_label,
        issue_label=issue_label,
        judge_label=judge_result.label,
        judge_rationale=judge_result.rationale,
        missing_facts=judge_result.missing_facts,
        contradictions=judge_result.contradictions,
        judge_model=judge_result.judge_model,
        judge_prompt_version=judge_result.judge_prompt_version,
        judge_latency_ms=judge_result.judge_latency_ms,
        judge_cost_usd=judge_result.judge_cost_usd,
        judge_unavailable=judge_result.judge_unavailable,
        source_recall=round(source_recall, 4),
        citation_matched=citation_matched,
        retrieved_expected_sources=retrieved_expected_sources,
        required_expected_sources=required_expected_sources,
        judge_source=judge_result.judge_source,
        deterministic_gate_failures=deterministic_gate_failures,
    )
