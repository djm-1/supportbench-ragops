from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from app.config import settings
from app.core.chunker import chunk_sections
from app.core.datasets import active_data_dir, dataset_status, eval_questions_path
from app.core.evaluator import evaluate_answer
from app.core.judge import judge_answer
from app.core.parser import parse_support_docs
from app.core.retriever import HybridRetriever
from app.core.types import EvalQuestion, RAGConfig, RetrievalCandidate, SupportChunk
from app.services.model_clients import ModelGateway


class RAGService:
    def __init__(self) -> None:
        self.chunks: list[SupportChunk] = []
        self.retriever: HybridRetriever | None = None
        self.model_client = ModelGateway()

    def ingest(self, data_dir: Path | None = None) -> tuple[int, int]:
        root = active_data_dir(data_dir or settings.data_dir)
        sections = parse_support_docs(root)
        chunk_kwargs = {"max_words": 900, "overlap_words": 0} if (root / "source_manifest.json").is_file() else {}
        self.chunks = chunk_sections(sections, **chunk_kwargs)
        self.retriever = HybridRetriever(self.chunks)
        documents = len({section["document"] for section in sections})
        return documents, len(self.chunks)

    def ensure_ready(self) -> None:
        if self.retriever is None:
            self.ingest()

    def load_eval_questions(
        self,
        limit: int | None = None,
        question_ids: list[str] | None = None,
    ) -> list[EvalQuestion]:
        payload = self.load_eval_question_payload()
        questions = [
            EvalQuestion(
                id=item["id"],
                question=item["question"],
                reference_answer=item["reference_answer"],
                expected_doc=item.get("expected_doc"),
                expected_section=item.get("expected_section"),
                tags=item.get("tags", []),
                should_refuse=item.get("should_refuse", False),
                question_type=item.get("question_type", "direct"),
                expected_sources=item.get("expected_sources", []),
                reference_facts=item.get("reference_facts", []),
                evaluation_notes=item.get("evaluation_notes", ""),
            )
            for item in payload
        ]
        if question_ids:
            selected = set(question_ids)
            questions = [question for question in questions if question.id in selected]
        return questions[:limit] if limit else questions

    def load_eval_question_payload(self) -> list[dict]:
        return json.loads(eval_questions_path(settings.data_dir).read_text(encoding="utf-8"))

    def dataset_status(self) -> dict:
        return dataset_status(settings.data_dir)

    def retrieve(self, question: str, config: RAGConfig) -> list[RetrievalCandidate]:
        self.ensure_ready()
        assert self.retriever is not None
        return self.retriever.retrieve(
            question,
            alpha=config.alpha,
            retrieve_top_k=config.retrieve_top_k,
            rerank_top_n=config.rerank_top_n,
        )

    def ask(self, question: str, config: RAGConfig) -> dict:
        candidates = self.retrieve(question, config)
        answer = self.model_client.generate(model=config.model, question=question, candidates=candidates)
        trace_id = f"trace_{uuid4().hex[:16]}"
        return {
            "trace_id": trace_id,
            "answer": answer,
            "candidates": candidates,
        }

    def evaluate_question(self, question: EvalQuestion, config: RAGConfig) -> dict:
        result = self.ask(question.question, config)
        judge_result = judge_answer(question, result["answer"], result["candidates"])
        metrics = evaluate_answer(question, result["answer"], result["candidates"], judge_result)
        return {
            **result,
            "question": question,
            "metrics": metrics,
        }


rag_service = RAGService()
