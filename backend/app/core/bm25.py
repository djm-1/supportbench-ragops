from __future__ import annotations

import math
from collections import Counter

from app.core.text import tokenize


class BM25Index:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.tokenized_docs = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.term_freqs = [Counter(doc) for doc in self.tokenized_docs]
        self.doc_freqs: Counter[str] = Counter()
        for doc in self.tokenized_docs:
            self.doc_freqs.update(set(doc))
        self.total_docs = len(documents)

    def idf(self, term: str) -> float:
        doc_freq = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def score(self, query: str, doc_index: int) -> float:
        query_terms = tokenize(query)
        if not query_terms:
            return 0.0
        score = 0.0
        term_freq = self.term_freqs[doc_index]
        doc_len = self.doc_lengths[doc_index] or 1
        for term in query_terms:
            freq = term_freq.get(term, 0)
            if freq == 0:
                continue
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += self.idf(term) * numerator / denominator
        return score

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores = [(index, self.score(query, index)) for index in range(len(self.documents))]
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:limit]
