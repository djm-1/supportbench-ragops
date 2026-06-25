from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings
from app.core.text import count_tokens, token_set
from app.core.types import AnswerResult, RetrievalCandidate


@dataclass(frozen=True)
class ModelProfile:
    alias: str
    display_name: str
    provider: str
    model_name: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    latency_ms: int
    quality_bias: float
    setup_command: str | None = None


MODEL_PROFILES: dict[str, ModelProfile] = {
    "groq_llama_3_1_8b": ModelProfile(
        "groq_llama_3_1_8b",
        "Groq Llama 3.1 8B",
        "Groq",
        "llama-3.1-8b-instant",
        0.0,
        0.0,
        650,
        0.84,
    ),
    "groq_llama_3_3_70b": ModelProfile(
        "groq_llama_3_3_70b",
        "Groq Llama 3.3 70B",
        "Groq",
        "llama-3.3-70b-versatile",
        0.0,
        0.0,
        950,
        0.89,
    ),
    "groq_gpt_oss_20b": ModelProfile(
        "groq_gpt_oss_20b",
        "Groq GPT OSS 20B",
        "Groq",
        "openai/gpt-oss-20b",
        0.0,
        0.0,
        750,
        0.86,
    ),
    "openai_primary": ModelProfile(
        "openai_primary",
        "OpenAI Primary",
        "OpenAI",
        settings.openai_model,
        0.00015,
        0.00060,
        1400,
        0.92,
    ),
    "gemini_flash": ModelProfile(
        "gemini_flash",
        "Gemini Flash",
        "Gemini",
        settings.gemini_model,
        0.000075,
        0.00030,
        1100,
        0.88,
    ),
}

DEFAULT_MODEL_ALIASES = list(MODEL_PROFILES.keys())


class DemoModelClient:
    """Deterministic local model simulator for tests and offline UI work."""

    def generate(
        self,
        *,
        model: str,
        question: str,
        candidates: list[RetrievalCandidate],
    ) -> AnswerResult:
        profile = MODEL_PROFILES.get(model, MODEL_PROFILES["openai_primary"])
        start = time.perf_counter()
        top = candidates[0] if candidates else None
        question_tokens = token_set(question)
        top_tokens = token_set((top.chunk.text + " " + top.chunk.section) if top else "")
        overlap = len(question_tokens & top_tokens) / max(1, len(question_tokens))
        fingerprint = int(hashlib.sha1(f"{model}:{question}".encode("utf-8")).hexdigest()[:8], 16)

        if top is None or overlap < 0.12:
            answer = "The provided support docs do not state enough information to answer this question."
            citations: list[dict[str, str]] = []
        else:
            answer = top.chunk.text
            citations = [
                {
                    "document": top.chunk.document,
                    "section": top.chunk.section,
                    "chunk_id": top.chunk.chunk_id,
                }
            ]

        if model in {"groq_gpt_oss_20b", "gemini_flash"} and fingerprint % 6 == 0 and citations:
            citations = []
        if model in {"groq_llama_3_1_8b", "gemini_flash"} and fingerprint % 5 == 0 and "except" in answer:
            answer = answer.split("except", 1)[0].strip() + "."
        if model == "openai_primary" and fingerprint % 13 == 0 and candidates:
            citations = [
                {
                    "document": candidates[-1].chunk.document,
                    "section": candidates[-1].chunk.section,
                    "chunk_id": candidates[-1].chunk.chunk_id,
                }
            ]
        if model == "groq_llama_3_3_70b" and fingerprint % 11 == 0:
            answer = answer + " Please verify this against the cited support article."

        context_text = "\n\n".join(candidate.chunk.text for candidate in candidates)
        input_tokens = count_tokens(question) + count_tokens(context_text)
        output_tokens = count_tokens(answer)
        cost = (input_tokens / 1000) * profile.input_cost_per_1k + (
            output_tokens / 1000
        ) * profile.output_cost_per_1k
        latency = profile.latency_ms + int((time.perf_counter() - start) * 1000) + (fingerprint % 180)

        return AnswerResult(
            answer=answer,
            citations=citations,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(cost, 6),
            latency_ms=latency,
            raw_response={"provider": "demo", "model_name": profile.model_name},
        )


class OpenAIModelClient:
    endpoint = "https://api.openai.com/v1/chat/completions"

    def generate(
        self,
        *,
        model: str,
        question: str,
        candidates: list[RetrievalCandidate],
    ) -> AnswerResult:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI benchmark model.")
        profile = MODEL_PROFILES[model]
        prompt = _build_prompt(question, candidates)
        start = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise customer-support RAG answerer. Never use outside knowledge.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 400,
        }
        with httpx.Client(timeout=90) as client:
            response = client.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return _answer_from_content(
            model=model,
            provider="openai",
            provider_model=settings.openai_model,
            profile=profile,
            question=question,
            candidates=candidates,
            content=content,
            usage={
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            },
            latency_ms=int((time.perf_counter() - start) * 1000),
            raw_response={"id": data.get("id"), "usage": usage},
        )


class GeminiModelClient:
    def generate(
        self,
        *,
        model: str,
        question: str,
        candidates: list[RetrievalCandidate],
    ) -> AnswerResult:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini benchmark model.")
        profile = MODEL_PROFILES[model]
        prompt = _build_prompt(question, candidates)
        start = time.perf_counter()
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemini_model}:generateContent"
        )
        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": "You are a precise customer-support RAG answerer. Never use outside knowledge."
                    }
                ]
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 400},
        }
        with httpx.Client(timeout=90) as client:
            response = client.post(endpoint, params={"key": settings.gemini_api_key}, json=payload)
            response.raise_for_status()
            data = response.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        content = "".join(str(part.get("text", "")) for part in parts).strip()
        usage = data.get("usageMetadata") or {}
        return _answer_from_content(
            model=model,
            provider="gemini",
            provider_model=settings.gemini_model,
            profile=profile,
            question=question,
            candidates=candidates,
            content=content,
            usage={
                "input_tokens": usage.get("promptTokenCount"),
                "output_tokens": usage.get("candidatesTokenCount"),
            },
            latency_ms=int((time.perf_counter() - start) * 1000),
            raw_response={"usage": usage},
        )


class GroqModelClient:
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def generate(
        self,
        *,
        model: str,
        question: str,
        candidates: list[RetrievalCandidate],
    ) -> AnswerResult:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required for Groq hosted open-source models.")
        profile = MODEL_PROFILES[model]
        prompt = _build_prompt(question, candidates)
        start = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": profile.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise customer-support RAG answerer. Never use outside knowledge.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 400,
        }
        with httpx.Client(timeout=90) as client:
            response = client.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return _answer_from_content(
            model=model,
            provider="groq",
            provider_model=profile.model_name,
            profile=profile,
            question=question,
            candidates=candidates,
            content=content,
            usage={
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            },
            latency_ms=int((time.perf_counter() - start) * 1000),
            raw_response={"id": data.get("id"), "usage": usage},
        )


class ModelGateway:
    def __init__(self) -> None:
        self.demo = DemoModelClient()
        self.openai = OpenAIModelClient()
        self.gemini = GeminiModelClient()
        self.groq = GroqModelClient()

    def generate(
        self,
        *,
        model: str,
        question: str,
        candidates: list[RetrievalCandidate],
    ) -> AnswerResult:
        if model not in MODEL_PROFILES:
            raise RuntimeError(f"Unknown model alias: {model}")
        if not settings.use_real_models:
            return self.demo.generate(model=model, question=question, candidates=candidates)
        provider = MODEL_PROFILES[model].provider
        if provider == "OpenAI":
            return self.openai.generate(model=model, question=question, candidates=candidates)
        if provider == "Gemini":
            return self.gemini.generate(model=model, question=question, candidates=candidates)
        if provider == "Groq":
            return self.groq.generate(model=model, question=question, candidates=candidates)
        raise RuntimeError(f"Unsupported provider: {provider}")


def model_catalog() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for profile in MODEL_PROFILES.values():
        available = True
        setup_message = ""
        if profile.provider == "OpenAI":
            available = bool(settings.openai_api_key)
            setup_message = "" if available else "Set OPENAI_API_KEY."
        elif profile.provider == "Gemini":
            available = bool(settings.gemini_api_key)
            setup_message = "" if available else "Set GEMINI_API_KEY."
        elif profile.provider == "Groq":
            available = bool(settings.groq_api_key)
            setup_message = "" if available else "Set GROQ_API_KEY."
        catalog.append(
            {
                "alias": profile.alias,
                "display_name": profile.display_name,
                "provider": profile.provider,
                "model": profile.model_name,
                "is_available": available,
                "requires": setup_message,
                "setup_command": profile.setup_command,
            }
        )
    return catalog


def _build_prompt(question: str, candidates: list[RetrievalCandidate]) -> str:
    context_payload = [
        {
            "chunk_id": candidate.chunk.chunk_id,
            "document": candidate.chunk.document,
            "section": candidate.chunk.section,
            "text": candidate.chunk.text,
        }
        for candidate in candidates
    ]
    return (
        "Answer the customer-support question using only the provided chunks. "
        "If the answer is not supported, say exactly: "
        "'The provided support docs do not state enough information to answer this question.' "
        "Return only JSON with keys answer and citations. citations must be a list of "
        "objects with document, section, and chunk_id.\n\n"
        f"Question: {question}\n\n"
        f"Chunks:\n{json.dumps(context_payload, indent=2)}"
    )


def _answer_from_content(
    *,
    model: str,
    provider: str,
    provider_model: str,
    profile: ModelProfile,
    question: str,
    candidates: list[RetrievalCandidate],
    content: str,
    usage: dict[str, Any],
    latency_ms: int,
    raw_response: dict[str, Any],
) -> AnswerResult:
    parsed = _parse_model_json(content)
    answer_text = str(parsed.get("answer") or content).strip()
    citations = parsed.get("citations") if isinstance(parsed.get("citations"), list) else []
    normalized_citations = [
        {
            "document": str(item.get("document", "")),
            "section": str(item.get("section", "")),
            "chunk_id": str(item.get("chunk_id", "")),
        }
        for item in citations
        if isinstance(item, dict)
    ]
    context_text = "\n\n".join(candidate.chunk.text for candidate in candidates)
    input_tokens = int(usage.get("input_tokens") or count_tokens(question) + count_tokens(context_text))
    output_tokens = int(usage.get("output_tokens") or count_tokens(answer_text))
    cost = (input_tokens / 1000) * profile.input_cost_per_1k + (
        output_tokens / 1000
    ) * profile.output_cost_per_1k
    return AnswerResult(
        answer=answer_text,
        citations=normalized_citations,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=round(cost, 8),
        latency_ms=latency_ms,
        raw_response={
            "provider": provider,
            "provider_model": provider_model,
            **raw_response,
        },
    )


def _parse_model_json(content: str) -> dict[str, Any]:
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
