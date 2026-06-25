from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from typing import Any

import httpx

from app.config import settings
from app.core.text import clamp, count_tokens, tokenize
from app.core.types import AnswerJudgeResult, AnswerResult, EvalQuestion, RetrievalCandidate


OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
OPENAI_RESPONSE_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "rationale": {"type": "string"},
        "missing_facts": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "label", "rationale", "missing_facts", "contradictions"],
    "additionalProperties": False,
}
CHAT_COMPLETIONS_JUDGE_PREFIXES = ("gpt-4o", "gpt-3.5", "o1", "o3")


def judge_answer(
    question: EvalQuestion,
    answer: AnswerResult,
    candidates: list[RetrievalCandidate],
) -> AnswerJudgeResult:
    fallback = deterministic_judge(question, answer)
    provider = settings.eval_judge_provider.lower().strip()
    if not _should_call_llm_judge(question, fallback):
        return fallback
    if provider in {"deterministic", "local", "off"} or not settings.use_real_models:
        return fallback

    providers = _judge_provider_order(provider)
    if not providers:
        return fallback

    errors: list[str] = []
    for candidate_provider in providers:
        try:
            if candidate_provider == "openai":
                return _openai_judge(question, answer, candidates)
            if candidate_provider == "gemini":
                return _gemini_judge(question, answer, candidates)
        except Exception as error:
            errors.append(f"{candidate_provider}: {error}")

    return AnswerJudgeResult(
        score=fallback.score,
        label="judge_unavailable",
        rationale=(
            "LLM judge failed for all configured providers, so deterministic factual scoring was used. "
            + " | ".join(errors)
        ),
        missing_facts=fallback.missing_facts,
        contradictions=fallback.contradictions,
        judge_model=", ".join(_judge_model_name(item) for item in providers),
        judge_prompt_version=settings.eval_judge_prompt_version,
        judge_unavailable=True,
        judge_source="fallback",
    )


def deterministic_judge(question: EvalQuestion, answer: AnswerResult) -> AnswerJudgeResult:
    answer_text = answer.answer
    if question.should_refuse:
        refused = _is_refusal(answer_text)
        return AnswerJudgeResult(
            score=1.0 if refused else 0.0,
            label="correct" if refused else "incorrect",
            rationale=(
                "The answer correctly refused unsupported information."
                if refused
                else "The question should be refused, but the answer attempted to provide unsupported information."
            ),
            contradictions=[] if refused else ["The answer should have refused this unsupported question."],
            judge_model="deterministic_fallback",
            judge_prompt_version=settings.eval_judge_prompt_version,
            judge_source="deterministic",
        )

    if _is_refusal(answer_text):
        return AnswerJudgeResult(
            score=0.0,
            label="incorrect",
            rationale="The answer refused even though the reference contains an answer.",
            missing_facts=["answerable_reference"],
            judge_model="deterministic_fallback",
            judge_prompt_version=settings.eval_judge_prompt_version,
            judge_source="deterministic",
        )

    reference = question.reference_answer
    reference_tokens = _normalized_tokens(reference)
    answer_tokens = _normalized_tokens(answer_text)
    lexical_f1 = _token_f1(reference_tokens, answer_tokens)
    sequence_score = SequenceMatcher(None, _canonical_text(reference), _canonical_text(answer_text)).ratio()
    semantic_proxy = max(lexical_f1, sequence_score)
    reference_hard_facts = _critical_facts(reference)
    answer_hard_facts = _critical_facts(answer_text)
    matched_hard_facts = [fact for fact in reference_hard_facts if fact in answer_hard_facts]
    missing_hard_facts = [fact for fact in reference_hard_facts if fact not in answer_hard_facts]
    missing_reference_facts = _missing_reference_fact_phrases(question, answer_text)
    contradictions = _hard_fact_contradictions(reference_hard_facts, answer_hard_facts)

    if contradictions:
        return AnswerJudgeResult(
            score=0.2,
            label="incorrect",
            rationale="The answer contradicts an objective fact from the reference answer.",
            missing_facts=missing_hard_facts + missing_reference_facts,
            contradictions=contradictions,
            judge_model="deterministic_fallback",
            judge_prompt_version=settings.eval_judge_prompt_version,
            judge_source="deterministic",
        )

    fact_score = len(matched_hard_facts) / len(reference_hard_facts) if reference_hard_facts else 1.0
    score = clamp((0.72 * semantic_proxy) + (0.28 * fact_score))

    if missing_hard_facts and not matched_hard_facts:
        score = min(score, 0.49)
    elif missing_hard_facts:
        score = min(score, 0.64)
    elif matched_hard_facts and semantic_proxy < 0.5:
        score = max(score, 0.66)

    if missing_reference_facts and score > 0.74:
        score = 0.74

    label = "correct" if score >= 0.78 else "partial" if score >= 0.55 else "incorrect"
    if matched_hard_facts and label == "incorrect":
        label = "partial"
        score = max(score, 0.58)

    if label == "correct":
        rationale = "The answer is factually close to the reference."
    elif matched_hard_facts:
        rationale = "The answer includes the key objective fact, but it is incomplete or too brief compared with the reference answer."
    elif label == "partial":
        rationale = "The answer overlaps with the reference but misses one or more important details."
    else:
        rationale = "The answer does not contain enough matching facts from the reference answer."

    return AnswerJudgeResult(
        score=round(score, 4),
        label=label,
        rationale=rationale,
        missing_facts=missing_hard_facts + missing_reference_facts,
        contradictions=contradictions,
        judge_model="deterministic_fallback",
        judge_prompt_version=settings.eval_judge_prompt_version,
        judge_source="deterministic",
    )


def _should_call_llm_judge(question: EvalQuestion, fallback: AnswerJudgeResult) -> bool:
    if question.should_refuse or question.question_type in {"missing_data", "missing", "refusal", "unsupported"}:
        return False
    if fallback.contradictions:
        return False
    if question.question_type in {"paraphrase", "calculation", "cross_document"}:
        return True
    return fallback.score < 0.93


def _openai_judge(
    question: EvalQuestion,
    answer: AnswerResult,
    candidates: list[RetrievalCandidate],
) -> AnswerJudgeResult:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured for answer judging.")
    model = settings.eval_judge_model or settings.openai_model
    if _uses_responses_api(model):
        return _openai_responses_judge(question, answer, candidates, model)
    return _openai_chat_judge(question, answer, candidates, model)


def _openai_chat_judge(
    question: EvalQuestion,
    answer: AnswerResult,
    candidates: list[RetrievalCandidate],
    model: str,
) -> AnswerJudgeResult:
    start = time.perf_counter()
    prompt = _judge_prompt(question, answer, candidates)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict factual-equivalence evaluator for customer-support RAG answers.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 350,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=90) as client:
        response = client.post(OPENAI_CHAT_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    return _judge_from_json(
        content,
        judge_model=f"openai:{model}",
        latency_ms=int((time.perf_counter() - start) * 1000),
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )


def _openai_responses_judge(
    question: EvalQuestion,
    answer: AnswerResult,
    candidates: list[RetrievalCandidate],
    model: str,
) -> AnswerJudgeResult:
    start = time.perf_counter()
    payload = {
        "model": model,
        "instructions": "You are a strict factual-equivalence evaluator for customer-support RAG answers.",
        "input": _judge_prompt(question, answer, candidates),
        "max_output_tokens": 350,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer_judge_result",
                "schema": OPENAI_RESPONSE_JUDGE_SCHEMA,
                "strict": True,
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=90) as client:
        response = client.post(OPENAI_RESPONSES_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    usage = data.get("usage") or {}
    return _judge_from_json(
        _extract_responses_text(data),
        judge_model=f"openai:{model}",
        latency_ms=int((time.perf_counter() - start) * 1000),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )


def _gemini_judge(
    question: EvalQuestion,
    answer: AnswerResult,
    candidates: list[RetrievalCandidate],
) -> AnswerJudgeResult:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured for answer judging.")
    model = settings.eval_judge_model or settings.gemini_model
    start = time.perf_counter()
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": "You are a strict factual-equivalence evaluator for customer-support RAG answers."
                }
            ]
        },
        "contents": [{"role": "user", "parts": [{"text": _judge_prompt(question, answer, candidates)}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 350,
            "responseMimeType": "application/json",
        },
    }
    with httpx.Client(timeout=90) as client:
        response = client.post(endpoint, params={"key": settings.gemini_api_key}, json=payload)
        response.raise_for_status()
        data = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    content = "".join(str(part.get("text", "")) for part in parts).strip()
    usage = data.get("usageMetadata") or {}
    return _judge_from_json(
        content,
        judge_model=f"gemini:{model}",
        latency_ms=int((time.perf_counter() - start) * 1000),
        input_tokens=usage.get("promptTokenCount"),
        output_tokens=usage.get("candidatesTokenCount"),
    )


def _judge_prompt(
    question: EvalQuestion,
    answer: AnswerResult,
    candidates: list[RetrievalCandidate],
) -> str:
    context = [
        {
            "document": candidate.chunk.document,
            "section": candidate.chunk.section,
            "text": candidate.chunk.text,
        }
        for candidate in candidates[:5]
    ]
    return (
        "Compare the generated answer to the reference answer for factual equivalence. "
        "Do not require identical wording. Penalize missing numbers, dates, fees, deadlines, URLs, email addresses, "
        "or contradictions. If the question should be refused, score only whether the answer refused unsupported info. "
        "Return JSON only with keys: score (0 to 1), label (correct, partial, incorrect), rationale, "
        "missing_facts (array), contradictions (array).\n\n"
        f"Question: {question.question}\n"
        f"Should refuse: {question.should_refuse}\n"
        f"Reference answer: {question.reference_answer}\n"
        f"Generated answer: {answer.answer}\n"
        f"Retrieved context: {json.dumps(context, ensure_ascii=False)}"
    )


def _judge_from_json(
    content: str,
    *,
    judge_model: str,
    latency_ms: int,
    input_tokens: Any,
    output_tokens: Any,
) -> AnswerJudgeResult:
    parsed = _parse_json_object(content)
    score = clamp(float(parsed.get("score", 0.0)))
    label = str(parsed.get("label") or ("correct" if score >= 0.78 else "partial" if score >= 0.55 else "incorrect"))
    if label not in {"correct", "partial", "incorrect"}:
        label = "partial"
    input_count = int(input_tokens or count_tokens(content))
    output_count = int(output_tokens or 0)
    return AnswerJudgeResult(
        score=round(score, 4),
        label=label,
        rationale=str(parsed.get("rationale") or ""),
        missing_facts=[str(item) for item in parsed.get("missing_facts", []) if item],
        contradictions=[str(item) for item in parsed.get("contradictions", []) if item],
        judge_model=judge_model,
        judge_prompt_version=settings.eval_judge_prompt_version,
        judge_latency_ms=latency_ms,
        judge_cost_usd=_estimate_judge_cost(judge_model, input_count, output_count),
        judge_source="llm",
    )


def _uses_responses_api(model: str) -> bool:
    return not model.startswith(CHAT_COMPLETIONS_JUDGE_PREFIXES)


def _extract_responses_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"])
    parts: list[str] = []
    for output in data.get("output", []) or []:
        for item in output.get("content", []) or []:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
    return "".join(parts).strip()


def _estimate_judge_cost(judge_model: str, input_tokens: int, output_tokens: int) -> float:
    model = judge_model.split(":", 1)[-1]
    input_per_million = 0.15
    output_per_million = 0.60
    if model == "gpt-5.4-nano":
        input_per_million = 0.20
        output_per_million = 1.25
    elif "mini" in model:
        input_per_million = 0.15
        output_per_million = 0.60
    return round(
        (input_tokens / 1_000_000) * input_per_million
        + (output_tokens / 1_000_000) * output_per_million,
        8,
    )


def _judge_provider_order(provider: str) -> list[str]:
    if provider == "auto":
        output: list[str] = []
        if settings.openai_api_key:
            output.append("openai")
        if settings.gemini_api_key:
            output.append("gemini")
        return output
    if provider in {"openai", "gemini"}:
        return [provider]
    return []


def _judge_model_name(provider: str) -> str:
    if provider == "openai":
        return f"openai:{settings.eval_judge_model or settings.openai_model}"
    if provider == "gemini":
        return f"gemini:{settings.eval_judge_model or settings.gemini_model}"
    return "deterministic_fallback"


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _normalized_tokens(text: str) -> set[str]:
    return {_stem(token) for token in tokenize(_canonical_text(text)) if len(token) > 2}


def _canonical_text(text: str) -> str:
    value = text.lower()
    replacements = {
        "oct.": "october",
        "oct ": "october ",
        "till": "until",
        "pickup": "pick up",
        "pick-up": "pick up",
        "cash on delivery": "cod",
        "rupees": "rs",
        "₹": "rs ",
        "inr": "rs",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _stem(token: str) -> str:
    token = token.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _token_f1(reference_tokens: set[str], answer_tokens: set[str]) -> float:
    if not reference_tokens or not answer_tokens:
        return 0.0
    overlap = len(reference_tokens & answer_tokens)
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _critical_facts(text: str) -> list[str]:
    value = _canonical_text(text)
    facts = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", value))
    facts.update(re.findall(r"\brs\s*\d+(?:\.\d+)?\b", value))
    facts.update(re.findall(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", value))
    facts.update(re.findall(r"https?://[^\s)]+", value))
    facts.update(re.findall(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", value))
    return sorted(facts)


def _missing_reference_fact_phrases(question: EvalQuestion, answer_text: str) -> list[str]:
    answer_tokens = _normalized_tokens(answer_text)
    answer_value = _canonical_text(answer_text)
    missing: list[str] = []
    for fact in question.reference_facts:
        normalized_fact = _canonical_text(str(fact)).strip()
        if not normalized_fact:
            continue
        if normalized_fact in answer_value:
            continue
        fact_tokens = _normalized_tokens(normalized_fact)
        if not fact_tokens:
            continue
        coverage = len(fact_tokens & answer_tokens) / len(fact_tokens)
        if coverage < 0.62:
            missing.append(str(fact))
    return missing


def _hard_fact_contradictions(reference_facts: list[str], answer_facts: list[str]) -> list[str]:
    reference_numbers = {fact for fact in reference_facts if re.search(r"\d", fact)}
    answer_numbers = {fact for fact in answer_facts if re.search(r"\d", fact)}
    if reference_numbers and answer_numbers and not reference_numbers & answer_numbers:
        return [
            "Expected one of "
            + ", ".join(reference_numbers)
            + " but the answer stated "
            + ", ".join(answer_numbers)
            + "."
        ]
    return []


def _is_refusal(text: str) -> bool:
    value = text.lower()
    return any(
        marker in value
        for marker in (
            "do not state",
            "do not specify",
            "do not mention",
            "not enough",
            "provided docs do not",
            "provided support docs do not",
            "cannot determine",
            "i don't know",
        )
    )
