from __future__ import annotations

import hashlib

from app.core.types import SupportChunk


def _chunk_id(document: str, section: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{document}:{section}:{index}:{text}".encode("utf-8")).hexdigest()
    return f"chunk_{digest[:12]}"


def chunk_sections(
    sections: list[dict[str, str]],
    *,
    max_words: int = 120,
    overlap_words: int = 24,
) -> list[SupportChunk]:
    chunks: list[SupportChunk] = []
    for section in sections:
        words = section["text"].split()
        if not words:
            continue
        step = max(1, max_words - overlap_words)
        windows = [words[i : i + max_words] for i in range(0, len(words), step)]
        for index, window in enumerate(windows):
            text = " ".join(window)
            chunks.append(
                SupportChunk(
                    chunk_id=_chunk_id(section["document"], section["section"], index, text),
                    document=section["document"],
                    section=section["section"],
                    text=text,
                    metadata={
                        "title": section.get("title", ""),
                        "chunk_index": index,
                        "word_count": len(window),
                    },
                )
            )
    return chunks
