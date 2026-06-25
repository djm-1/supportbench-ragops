from __future__ import annotations

import time
import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings
from app.core.embeddings import HashEmbeddingModel, cosine_similarity
from app.core.types import SupportChunk


class DenseIndex:
    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        raise NotImplementedError


class LocalDenseIndex(DenseIndex):
    def __init__(self, chunks: list[SupportChunk]) -> None:
        self.chunks = chunks
        self.embedding_model = HashEmbeddingModel()
        self.chunk_embeddings = [self.embedding_model.embed(chunk.text) for chunk in chunks]

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        query_embedding = self.embedding_model.embed(query)
        dense_scores = [
            (chunk.chunk_id, cosine_similarity(query_embedding, chunk_embedding))
            for chunk, chunk_embedding in zip(self.chunks, self.chunk_embeddings)
        ]
        dense_scores.sort(key=lambda item: item[1], reverse=True)
        return dense_scores[:limit]


class OpenAIEmbeddingClient:
    endpoint = "https://api.openai.com/v1/embeddings"

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "PINECONE_EMBEDDING_MODE=openai requires OPENAI_API_KEY for embeddings."
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {
            "model": settings.embedding_model,
            "input": texts,
            "dimensions": settings.embedding_dimension,
        }
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=90) as client:
            response = client.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        ordered = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


@dataclass
class PineconeIndexInfo:
    name: str
    host: str
    dimension: int | None
    metric: str
    embed: dict[str, Any] | None = None

    @property
    def uses_integrated_embedding(self) -> bool:
        return bool(self.embed)


class PineconeRestClient:
    control_plane = "https://api.pinecone.io"
    api_version = "2025-10"
    search_api_version = "unstable"

    def __init__(self) -> None:
        if not settings.pinecone_api_key:
            raise RuntimeError("USE_PINECONE=true requires PINECONE_API_KEY.")
        self.headers = {
            "Api-Key": settings.pinecone_api_key,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": self.api_version,
        }

    def ensure_index(self) -> PineconeIndexInfo:
        existing = self.describe_index()
        if existing:
            return existing

        payload = {
            "name": settings.pinecone_index,
            "dimension": settings.embedding_dimension,
            "metric": settings.pinecone_metric,
            "spec": {
                "serverless": {
                    "cloud": settings.pinecone_cloud,
                    "region": settings.pinecone_region,
                }
            },
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.control_plane}/indexes", headers=self.headers, json=payload
            )
            response.raise_for_status()

        deadline = time.time() + 120
        while time.time() < deadline:
            info = self.describe_index()
            if info and info.host:
                return info
            time.sleep(2)
        raise RuntimeError(f"Pinecone index {settings.pinecone_index} was created but not ready.")

    def describe_index(self) -> PineconeIndexInfo | None:
        with httpx.Client(timeout=60) as client:
            response = client.get(
                f"{self.control_plane}/indexes/{settings.pinecone_index}",
                headers=self.headers,
            )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        return PineconeIndexInfo(
            name=data["name"],
            host=data["host"],
            dimension=int(data["dimension"]) if data.get("dimension") else None,
            metric=data.get("metric") or settings.pinecone_metric,
            embed=data.get("embed"),
        )

    def ensure_integrated_index(self) -> PineconeIndexInfo:
        existing = self.describe_index()
        if existing:
            if not existing.uses_integrated_embedding:
                raise RuntimeError(
                    f"PINECONE_INDEX={settings.pinecone_index!r} already exists as a standard "
                    "vector index. Pinecone hosted embeddings require an integrated index. "
                    "Set PINECONE_INDEX to a fresh name like 'supportbench-ragops-integrated', "
                    "or delete/recreate the existing index."
                )
            field_map = (existing.embed or {}).get("field_map") or {}
            if field_map.get("text") != settings.pinecone_text_field:
                raise RuntimeError(
                    f"Pinecone integrated index field_map text={field_map.get('text')!r} does "
                    f"not match PINECONE_TEXT_FIELD={settings.pinecone_text_field!r}."
                )
            return existing

        payload = {
            "name": settings.pinecone_index,
            "cloud": settings.pinecone_cloud,
            "region": settings.pinecone_region,
            "embed": {
                "model": settings.pinecone_embed_model,
                "field_map": {"text": settings.pinecone_text_field},
            },
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.control_plane}/indexes/create-for-model",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()

        deadline = time.time() + 120
        while time.time() < deadline:
            info = self.describe_index()
            if info and info.host:
                return info
            time.sleep(2)
        raise RuntimeError(f"Pinecone index {settings.pinecone_index} was created but not ready.")

    def clear_namespace(self, host: str) -> None:
        payload = {"namespace": settings.pinecone_namespace, "deleteAll": True}
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"https://{host}/vectors/delete",
                headers=self.headers,
                json=payload,
            )
        if response.status_code not in {200, 202, 204, 404}:
            response.raise_for_status()

    def upsert(self, host: str, vectors: list[dict[str, Any]]) -> None:
        if not vectors:
            return
        with httpx.Client(timeout=90) as client:
            for index in range(0, len(vectors), settings.pinecone_batch_size):
                payload = {
                    "namespace": settings.pinecone_namespace,
                    "vectors": vectors[index : index + settings.pinecone_batch_size],
                }
                response = client.post(
                    f"https://{host}/vectors/upsert",
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()

    def query(self, host: str, vector: list[float], limit: int) -> list[tuple[str, float]]:
        payload = {
            "namespace": settings.pinecone_namespace,
            "vector": vector,
            "topK": limit,
            "includeMetadata": True,
            "includeValues": False,
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(f"https://{host}/query", headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return [(match["id"], float(match.get("score") or 0.0)) for match in data.get("matches", [])]

    def upsert_records(self, host: str, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        headers = {
            **self.headers,
            "Content-Type": "application/x-ndjson",
        }
        batch_size = min(settings.pinecone_batch_size, 96)
        with httpx.Client(timeout=90) as client:
            for index in range(0, len(records), batch_size):
                batch = records[index : index + batch_size]
                body = "\n".join(json.dumps(record) for record in batch)
                response = client.post(
                    f"https://{host}/records/namespaces/{settings.pinecone_namespace}/upsert",
                    headers=headers,
                    content=body,
                )
                response.raise_for_status()

    def search_records(self, host: str, query: str, limit: int) -> list[tuple[str, float]]:
        headers = {
            **self.headers,
            "X-Pinecone-API-Version": self.search_api_version,
        }
        payload = {
            "query": {
                "inputs": {"text": query},
                "top_k": limit,
            },
            "fields": ["document", "section", settings.pinecone_text_field],
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"https://{host}/records/namespaces/{settings.pinecone_namespace}/search",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        hits = data.get("result", {}).get("hits", [])
        return [
            (hit.get("_id") or hit.get("id"), float(hit.get("_score") or hit.get("score") or 0.0))
            for hit in hits
            if hit.get("_id") or hit.get("id")
        ]


class PineconeDenseIndex(DenseIndex):
    def __init__(self, chunks: list[SupportChunk]) -> None:
        self.chunks = chunks
        self.embedding_client = OpenAIEmbeddingClient()
        self.pinecone = PineconeRestClient()
        info = self.pinecone.ensure_index()
        if info.uses_integrated_embedding:
            raise RuntimeError(
                f"PINECONE_INDEX={settings.pinecone_index!r} is an integrated embedding index. "
                "Use PINECONE_EMBEDDING_MODE=integrated with this index."
            )
        if info.dimension != settings.embedding_dimension:
            raise RuntimeError(
                f"Pinecone index dimension {info.dimension} does not match "
                f"EMBEDDING_DIMENSION={settings.embedding_dimension}."
            )
        self.host = info.host
        self._upsert_chunks()

    def _upsert_chunks(self) -> None:
        self.pinecone.clear_namespace(self.host)
        texts = [chunk.text for chunk in self.chunks]
        vectors: list[dict[str, Any]] = []
        for offset in range(0, len(texts), settings.embedding_batch_size):
            batch_chunks = self.chunks[offset : offset + settings.embedding_batch_size]
            batch_texts = texts[offset : offset + settings.embedding_batch_size]
            embeddings = self.embedding_client.embed_texts(batch_texts)
            for chunk, embedding in zip(batch_chunks, embeddings):
                vectors.append(
                    {
                        "id": chunk.chunk_id,
                        "values": embedding,
                        "metadata": {
                            "document": chunk.document,
                            "section": chunk.section,
                            "text": chunk.text,
                            **chunk.metadata,
                        },
                    }
                )
        self.pinecone.upsert(self.host, vectors)

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        query_embedding = self.embedding_client.embed_text(query)
        return self.pinecone.query(self.host, query_embedding, limit=limit)


class PineconeIntegratedDenseIndex(DenseIndex):
    def __init__(self, chunks: list[SupportChunk]) -> None:
        self.chunks = chunks
        self.pinecone = PineconeRestClient()
        info = self.pinecone.ensure_integrated_index()
        self.host = info.host
        self._upsert_chunks()

    def _upsert_chunks(self) -> None:
        records = [
            {
                "_id": chunk.chunk_id,
                settings.pinecone_text_field: chunk.text,
                "document": chunk.document,
                "section": chunk.section,
                **chunk.metadata,
            }
            for chunk in self.chunks
        ]
        self.pinecone.upsert_records(self.host, records)

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        return self.pinecone.search_records(self.host, query, limit=limit)


def build_dense_index(chunks: list[SupportChunk]) -> DenseIndex:
    if settings.use_pinecone:
        if settings.pinecone_embedding_mode.lower() == "integrated":
            return PineconeIntegratedDenseIndex(chunks)
        if settings.pinecone_embedding_mode.lower() != "openai":
            raise RuntimeError(
                "PINECONE_EMBEDDING_MODE must be 'integrated' or 'openai'."
            )
        return PineconeDenseIndex(chunks)
    return LocalDenseIndex(chunks)
