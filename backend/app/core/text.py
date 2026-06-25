from __future__ import annotations

import re
from collections import Counter

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def token_counter(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def count_tokens(text: str) -> int:
    return len(tokenize(text, keep_stopwords=True))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def overlap_ratio(candidate: str, reference: str) -> float:
    candidate_tokens = token_set(candidate)
    reference_tokens = token_set(reference)
    if not reference_tokens:
        return 0.0
    return len(candidate_tokens & reference_tokens) / len(reference_tokens)
