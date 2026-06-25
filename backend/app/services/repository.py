from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean
from typing import Any, Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db_models import (
    ChunkRecord,
    DashboardStateRecord,
    EvalQuestionRecord,
    ExperimentResultRecord,
    ExperimentRunRecord,
    TraceRecord,
)
from app.services.model_clients import DEFAULT_MODEL_ALIASES, MODEL_PROFILES


class ExperimentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def replace_chunks(self, chunks: list[dict[str, Any]]) -> None:
        self.db.execute(delete(ChunkRecord))
        for chunk in chunks:
            self.db.add(
                ChunkRecord(
                    chunk_id=chunk["chunk_id"],
                    document=chunk["document"],
                    section=chunk["section"],
                    text=chunk["text"],
                    metadata_json=chunk.get("metadata", {}),
                )
            )
        self.db.commit()

    def replace_eval_questions(self, questions: list[dict[str, Any]]) -> None:
        self.db.execute(delete(EvalQuestionRecord))
        for question in questions:
            self.db.add(
                EvalQuestionRecord(
                    id=question["id"],
                    question=question["question"],
                    reference_answer=question["reference_answer"],
                    expected_doc=question.get("expected_doc"),
                    expected_section=question.get("expected_section"),
                    tags=question.get("tags", []),
                    should_refuse=question.get("should_refuse", False),
                    question_type=question.get("question_type", "direct"),
                    expected_sources=question.get("expected_sources", []),
                    reference_facts=question.get("reference_facts", []),
                    evaluation_notes=question.get("evaluation_notes", ""),
                )
            )
        self.db.commit()

    def create_run(self, run_name: str, config_grid: dict[str, Any]) -> ExperimentRunRecord:
        record = ExperimentRunRecord(run_name=run_name, config_grid_json=config_grid, status="running")
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        self.start_run(record.id)
        return record

    def _dashboard_state(self) -> DashboardStateRecord:
        state = self.db.get(DashboardStateRecord, 1)
        if state:
            return state
        state = DashboardStateRecord(id=1, dashboard_state="empty")
        self.db.add(state)
        self.db.commit()
        self.db.refresh(state)
        return state

    def clear_active_dashboard(self) -> None:
        state = self._dashboard_state()
        state.active_run_id = None
        state.dashboard_state = "empty"
        state.last_ingested_at = datetime.utcnow()
        state.updated_at = datetime.utcnow()
        self.db.commit()

    def start_run(self, run_id: int) -> None:
        state = self._dashboard_state()
        if state.dashboard_state != "completed":
            state.active_run_id = run_id
            state.dashboard_state = "running"
            state.updated_at = datetime.utcnow()
            self.db.commit()

    def add_result(
        self,
        *,
        run_id: int,
        question_id: str,
        model: str,
        alpha: float,
        retrieve_top_k: int,
        rerank_top_n: int,
        answer_json: dict[str, Any],
        retrieved_json: list[dict[str, Any]],
        metrics_json: dict[str, Any],
        trace_id: str,
    ) -> None:
        question = self.db.get(EvalQuestionRecord, question_id)
        question_payload = _question_payload(question)
        self.db.add(
            ExperimentResultRecord(
                run_id=run_id,
                question_id=question_id,
                model=model,
                alpha=alpha,
                retrieve_top_k=retrieve_top_k,
                rerank_top_n=rerank_top_n,
                quality_score=float(metrics_json["quality_score"]),
                cost_usd=float(answer_json["estimated_cost_usd"]),
                latency_ms=int(answer_json["latency_ms"]),
                failure_category=str(metrics_json["failure_category"]),
                answer_json=answer_json,
                retrieved_json=retrieved_json,
                metrics_json=metrics_json,
                trace_id=trace_id,
            )
        )
        self.db.add(
            TraceRecord(
                trace_id=trace_id,
                run_id=run_id,
                question_id=question_id,
                payload_json={
                    "question_id": question_id,
                    **question_payload,
                    "model": model,
                    "alpha": alpha,
                    "retrieve_top_k": retrieve_top_k,
                    "rerank_top_n": rerank_top_n,
                    "answer": answer_json,
                    "retrieved_chunks": retrieved_json,
                    "metrics": metrics_json,
                },
            )
        )

    def complete_run(self, run_id: int, summary: dict[str, Any]) -> None:
        run = self.db.get(ExperimentRunRecord, run_id)
        if not run:
            return
        run.status = "completed"
        run.completed_at = datetime.utcnow()
        run.summary_json = summary
        state = self._dashboard_state()
        state.active_run_id = run_id
        state.dashboard_state = "completed"
        state.updated_at = datetime.utcnow()
        self.db.commit()

    def fail_run(self, run_id: int, message: str) -> None:
        run = self.db.get(ExperimentRunRecord, run_id)
        if not run:
            return
        run.status = "failed"
        run.completed_at = datetime.utcnow()
        run.summary_json = {"error": message}
        state = self._dashboard_state()
        if state.active_run_id == run_id:
            state.dashboard_state = "failed"
            state.updated_at = datetime.utcnow()
        self.db.commit()

    def summarize_run(self, run_id: int) -> dict[str, Any]:
        results = list(
            self.db.scalars(
                select(ExperimentResultRecord).where(ExperimentResultRecord.run_id == run_id)
            )
        )
        return summarize_results(results)

    def list_runs(self) -> list[dict[str, Any]]:
        runs = list(
            self.db.scalars(select(ExperimentRunRecord).order_by(ExperimentRunRecord.started_at.desc()))
        )
        return [
            {
                "id": run.id,
                "run_name": run.run_name,
                "status": run.status,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "summary": run.summary_json,
            }
            for run in runs
        ]

    def get_run(self, run_id: int) -> Optional[dict[str, Any]]:
        run = self.db.get(ExperimentRunRecord, run_id)
        if not run:
            return None
        results = list(
            self.db.scalars(
                select(ExperimentResultRecord).where(ExperimentResultRecord.run_id == run_id)
            )
        )
        state = self._dashboard_state()
        question_bank_changed = bool(
            state.last_ingested_at
            and run.completed_at
            and state.last_ingested_at > run.completed_at
        )
        traces = {
            trace.trace_id: trace.payload_json
            for trace in self.db.scalars(
                select(TraceRecord).where(TraceRecord.run_id == run_id)
            ).all()
        }
        questions = {
            question.id: question
            for question in self.db.scalars(select(EvalQuestionRecord)).all()
        }
        return {
            "id": run.id,
            "run_name": run.run_name,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "config_grid": run.config_grid_json,
            "summary": run.summary_json,
            "results": [
                serialize_result(
                    result,
                    None
                    if question_bank_changed and not (traces.get(result.trace_id) or {}).get("question")
                    else questions.get(result.question_id),
                    traces.get(result.trace_id),
                )
                for result in results
            ],
        }

    def export_run_csv(self, run_id: int) -> Optional[str]:
        run = self.get_run(run_id)
        if not run:
            return None

        output = io.StringIO()
        fieldnames = [
            "run_id",
            "run_name",
            "question_id",
            "question",
            "reference_answer",
            "model",
            "alpha",
            "retrieve_top_k",
            "rerank_top_n",
            "generated_answer",
            "quality_score",
            "answer_correctness",
            "groundedness",
            "citation_accuracy",
            "refusal_correctness",
            "expected_doc",
            "expected_section",
            "top_retrieved_doc",
            "top_retrieved_section",
            "expected_source_found",
            "question_type",
            "expected_sources",
            "source_recall",
            "citation_matched",
            "retrieved_expected_sources",
            "required_expected_sources",
            "judge_source",
            "deterministic_gate_failures",
            "failure_category",
            "judge_label",
            "judge_rationale",
            "missing_facts",
            "contradictions",
            "judge_model",
            "judge_prompt_version",
            "latency_ms",
            "cost_usd",
            "trace_id",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in run["results"]:
            metrics = result.get("metrics") or {}
            writer.writerow(
                {
                    "run_id": run["id"],
                    "run_name": run["run_name"],
                    "question_id": result["question_id"],
                    "question": result.get("question", ""),
                    "reference_answer": result.get("reference_answer", ""),
                    "model": result["model"],
                    "alpha": result["alpha"],
                    "retrieve_top_k": result["retrieve_top_k"],
                    "rerank_top_n": result["rerank_top_n"],
                    "generated_answer": (result.get("answer") or {}).get("answer", ""),
                    "quality_score": result["quality_score"],
                    "answer_correctness": metrics.get("answer_correctness", ""),
                    "groundedness": metrics.get("groundedness", ""),
                    "citation_accuracy": metrics.get("citation_accuracy", ""),
                    "refusal_correctness": metrics.get("refusal_correctness", ""),
                    "expected_doc": result.get("expected_doc", ""),
                    "expected_section": result.get("expected_section", ""),
                    "top_retrieved_doc": result.get("top_retrieved_doc", ""),
                    "top_retrieved_section": result.get("top_retrieved_section", ""),
                    "expected_source_found": result.get("expected_source_found", False),
                    "question_type": result.get("question_type", ""),
                    "expected_sources": json.dumps(result.get("expected_sources", []), ensure_ascii=False),
                    "source_recall": metrics.get("source_recall", ""),
                    "citation_matched": metrics.get("citation_matched", ""),
                    "retrieved_expected_sources": json.dumps(
                        metrics.get("retrieved_expected_sources", []), ensure_ascii=False
                    ),
                    "required_expected_sources": json.dumps(
                        metrics.get("required_expected_sources", []), ensure_ascii=False
                    ),
                    "judge_source": metrics.get("judge_source", ""),
                    "deterministic_gate_failures": "; ".join(
                        metrics.get("deterministic_gate_failures", []) or []
                    ),
                    "failure_category": result["failure_category"],
                    "judge_label": metrics.get("judge_label", ""),
                    "judge_rationale": metrics.get("judge_rationale", ""),
                    "missing_facts": "; ".join(metrics.get("missing_facts", []) or []),
                    "contradictions": "; ".join(metrics.get("contradictions", []) or []),
                    "judge_model": metrics.get("judge_model", ""),
                    "judge_prompt_version": metrics.get("judge_prompt_version", ""),
                    "latency_ms": result["latency_ms"],
                    "cost_usd": result["cost_usd"],
                    "trace_id": result["trace_id"],
                }
            )
        return output.getvalue()

    def dashboard_summary(self) -> dict[str, Any]:
        state = self._dashboard_state()
        last_ingested_at = state.last_ingested_at.isoformat() if state.last_ingested_at else None
        display_run: ExperimentRunRecord | None = None

        if state.active_run_id is not None:
            active_run = self.db.get(ExperimentRunRecord, state.active_run_id)
            if active_run and active_run.status == "completed" and _uses_current_models(active_run):
                display_run = active_run

        if display_run is None:
            display_run = self._latest_completed_run()

        if display_run is None:
            return empty_summary(
                dashboard_state=state.dashboard_state,
                last_ingested_at=last_ingested_at,
                summary_notice="No completed benchmark is available yet.",
            )

        results = list(
            self.db.scalars(
                select(ExperimentResultRecord).where(ExperimentResultRecord.run_id == display_run.id)
            )
        )
        summary = summarize_results(results)
        summary["run_id"] = display_run.id
        summary["run_name"] = display_run.run_name
        summary["dashboard_state"] = state.dashboard_state
        summary["last_ingested_at"] = last_ingested_at
        summary["is_stale"] = not (
            state.dashboard_state == "completed" and state.active_run_id == display_run.id
        )
        summary["summary_notice"] = _summary_notice(state, display_run)
        return summary

    def _latest_completed_run(self) -> ExperimentRunRecord | None:
        runs = list(
            self.db.scalars(
                select(ExperimentRunRecord)
                .where(ExperimentRunRecord.status == "completed")
                .order_by(ExperimentRunRecord.id.desc())
            )
        )
        for run in runs:
            if _uses_current_models(run):
                return run
        return None

    def dashboard_history(self) -> dict[str, Any]:
        runs = list(
            self.db.scalars(
                select(ExperimentRunRecord)
                .where(ExperimentRunRecord.status == "completed")
                .order_by(ExperimentRunRecord.started_at.asc())
            )
        )
        items = []
        current_aliases = set(DEFAULT_MODEL_ALIASES)
        valid_run_ids: list[int] = []
        for run in runs:
            configured_models = set((run.config_grid_json or {}).get("models", []))
            if configured_models and not configured_models <= current_aliases:
                continue
            valid_run_ids.append(run.id)
            summary = run.summary_json or {}
            best = summary.get("best_config") or {}
            items.append(
                {
                    "run_id": run.id,
                    "run_name": run.run_name,
                    "started_at": run.started_at.isoformat() if run.started_at else None,
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                    "best_model": best.get("model"),
                    "best_quality": best.get("quality_score", 0),
                    "cost_per_query": best.get("cost_per_query", 0),
                    "avg_latency_ms": best.get("avg_latency_ms", 0),
                    "failure_rate": best.get("failure_rate", 0),
                    "result_count": summary.get("result_count", 0),
                }
            )

        results: list[ExperimentResultRecord] = []
        for run_id in valid_run_ids:
            results.extend(
                self.db.scalars(
                    select(ExperimentResultRecord).where(ExperimentResultRecord.run_id == run_id)
                )
            )
        results = [result for result in results if result.model in current_aliases]

        return {
            "runs": items,
            "model_performance": aggregate_history_results(results, by_config=False),
            "config_performance": aggregate_history_results(results, by_config=True),
        }

    def get_trace(self, trace_id: str) -> Optional[dict[str, Any]]:
        trace = self.db.get(TraceRecord, trace_id)
        if not trace:
            return None
        return {
            "trace_id": trace.trace_id,
            "run_id": trace.run_id,
            "question_id": trace.question_id,
            "created_at": trace.created_at.isoformat() if trace.created_at else None,
            "payload": trace.payload_json,
        }


def serialize_result(
    result: ExperimentResultRecord,
    question: EvalQuestionRecord | None = None,
    trace_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    top_retrieved = result.retrieved_json[0] if result.retrieved_json else {}
    question_payload = _question_payload(question)
    if trace_payload:
        for key in question_payload:
            if key in trace_payload:
                question_payload[key] = trace_payload[key]
    expected_doc = question_payload["expected_doc"]
    expected_section = question_payload["expected_section"]
    expected_sources = question_payload["expected_sources"] or (
        [{"document": expected_doc, "section": expected_section}]
        if expected_doc and expected_section
        else []
    )
    question_payload["expected_sources"] = expected_sources
    expected_source_found = bool(result.metrics_json.get("source_recall") == 1)
    if expected_sources and "source_recall" not in result.metrics_json:
        expected_source_found = all(
            any(
                chunk.get("document") == source.get("document")
                and chunk.get("section") == source.get("section")
                for chunk in result.retrieved_json
            )
            for source in expected_sources
        )

    payload = {
        "id": result.id,
        "question_id": result.question_id,
        "model": result.model,
        "alpha": result.alpha,
        "retrieve_top_k": result.retrieve_top_k,
        "rerank_top_n": result.rerank_top_n,
        "quality_score": result.quality_score,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
        "failure_category": result.failure_category,
        "result_label": result.metrics_json.get("result_label", _legacy_result_label(result.failure_category)),
        "issue_label": result.metrics_json.get("issue_label", _legacy_issue_label(result.failure_category)),
        "trace_id": result.trace_id,
        "metrics": result.metrics_json,
        "answer": result.answer_json,
        "retrieved_chunks": result.retrieved_json,
        **question_payload,
        "expected_doc": expected_doc,
        "expected_section": expected_section,
        "top_retrieved_doc": top_retrieved.get("document"),
        "top_retrieved_section": top_retrieved.get("section"),
        "expected_source_found": expected_source_found,
        "source_recall": result.metrics_json.get("source_recall", 1.0 if expected_source_found else 0.0),
        "citation_matched": result.metrics_json.get("citation_matched", result.metrics_json.get("citation_accuracy") == 1),
        "judge_label": result.metrics_json.get("judge_label", "not_run"),
        "judge_rationale": result.metrics_json.get("judge_rationale", ""),
        "judge_source": result.metrics_json.get("judge_source", "deterministic"),
        "missing_facts": result.metrics_json.get("missing_facts", []),
        "contradictions": result.metrics_json.get("contradictions", []),
        "judge_model": result.metrics_json.get("judge_model", ""),
        "judge_prompt_version": result.metrics_json.get("judge_prompt_version", ""),
    }
    return payload


def _legacy_result_label(failure_category: str) -> str:
    if failure_category == "passed":
        return "right"
    if failure_category in {"partial_answer", "partial_citation", "missing_citation", "needs_review"}:
        return "partial"
    return "wrong"


def _legacy_issue_label(failure_category: str) -> str:
    if failure_category == "passed":
        return "none"
    if failure_category == "bad_retrieval":
        return "source_not_found"
    if failure_category in {"missing_citation", "partial_citation", "wrong_citation", "unsupported_citation"}:
        return "citation_issue"
    if failure_category in {"partial_answer", "needs_review", "incorrect_refusal"}:
        return "incomplete_answer"
    return "unsupported_answer"


def _question_payload(question: EvalQuestionRecord | None) -> dict[str, Any]:
    if not question:
        return {
            "question": "Question snapshot unavailable for this legacy run.",
            "reference_answer": "Reference answer snapshot unavailable for this legacy run.",
            "expected_doc": None,
            "expected_section": None,
            "expected_sources": [],
            "question_type": "unknown",
            "reference_facts": [],
            "evaluation_notes": "",
            "tags": [],
            "should_refuse": False,
        }
    expected_sources = [
        {"document": str(source.get("document")), "section": str(source.get("section"))}
        for source in (question.expected_sources or [])
        if source.get("document") and source.get("section")
    ]
    if not expected_sources and question.expected_doc and question.expected_section:
        expected_sources = [{"document": question.expected_doc, "section": question.expected_section}]
    return {
        "question": question.question,
        "reference_answer": question.reference_answer,
        "expected_doc": question.expected_doc,
        "expected_section": question.expected_section,
        "expected_sources": expected_sources,
        "question_type": question.question_type or "direct",
        "reference_facts": question.reference_facts or [],
        "evaluation_notes": question.evaluation_notes or "",
        "tags": question.tags or [],
        "should_refuse": question.should_refuse,
    }


def aggregate_history_results(
    results: list[ExperimentResultRecord],
    *,
    by_config: bool,
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[ExperimentResultRecord]] = defaultdict(list)
    for result in results:
        key = (
            (result.model, result.alpha, result.retrieve_top_k, result.rerank_top_n)
            if by_config
            else result.model
        )
        grouped[key].append(result)

    rows: list[dict[str, Any]] = []
    for key, grouped_results in grouped.items():
        model = key[0] if by_config else key
        metrics = [result.metrics_json for result in grouped_results]
        row = {
            "model": model,
            "provider": MODEL_PROFILES.get(model).provider if model in MODEL_PROFILES else "Unknown",
            "run_count": len({result.run_id for result in grouped_results if result.run_id is not None}),
            "result_count": len(grouped_results),
            "avg_quality": round(mean(result.quality_score for result in grouped_results), 4),
            "avg_cost": round(mean(result.cost_usd for result in grouped_results), 6),
            "avg_latency": round(mean(result.latency_ms for result in grouped_results)),
            "failure_rate": round(
                sum(1 for result in grouped_results if result.failure_category != "passed")
                / max(1, len(grouped_results)),
                4,
            ),
            "answer_correctness": round(mean(metric["answer_correctness"] for metric in metrics), 4),
            "groundedness": round(mean(metric["groundedness"] for metric in metrics), 4),
            "citation_accuracy": round(mean(metric["citation_accuracy"] for metric in metrics), 4),
            "refusal_correctness": round(mean(metric["refusal_correctness"] for metric in metrics), 4),
        }
        if by_config:
            row.update(
                {
                    "alpha": key[1],
                    "retrieve_top_k": key[2],
                    "rerank_top_n": key[3],
                }
            )
        rows.append(row)

    rows.sort(key=lambda item: (item["avg_quality"], -item["avg_cost"]), reverse=True)
    return rows


def empty_summary(
    *,
    dashboard_state: str = "empty",
    last_ingested_at: str | None = None,
    summary_notice: str | None = None,
) -> dict[str, Any]:
    return {
        "dashboard_state": dashboard_state,
        "last_ingested_at": last_ingested_at,
        "summary_notice": summary_notice,
        "is_stale": False,
        "best_config": None,
        "best_model": None,
        "cheapest_acceptable_model": None,
        "fastest_acceptable_model": None,
        "most_common_failure": None,
        "leaderboard": [],
        "config_leaderboard": [],
        "model_leaderboard": [],
        "quality_cost_points": [],
        "failure_breakdown": [],
        "failure_by_model": [],
        "model_summary": {},
        "result_count": 0,
    }


def _uses_current_models(run: ExperimentRunRecord) -> bool:
    configured_models = set((run.config_grid_json or {}).get("models", []))
    return not configured_models or configured_models <= set(DEFAULT_MODEL_ALIASES)


def _summary_notice(state: DashboardStateRecord, display_run: ExperimentRunRecord) -> str:
    if state.dashboard_state == "failed":
        return "Latest benchmark failed; showing previous successful run."
    if state.dashboard_state == "empty" and state.last_ingested_at:
        return "Docs were re-ingested; run a new benchmark to refresh this decision."
    if state.dashboard_state == "running":
        return "Benchmark running; showing latest completed benchmark."
    if state.dashboard_state == "completed" and state.active_run_id == display_run.id:
        return "Showing latest completed benchmark."
    return "Showing latest completed benchmark."


def summarize_results(results: list[ExperimentResultRecord]) -> dict[str, Any]:
    grouped: dict[tuple[str, float, int, int], list[ExperimentResultRecord]] = defaultdict(list)
    for result in results:
        grouped[(result.model, result.alpha, result.retrieve_top_k, result.rerank_top_n)].append(result)

    config_leaderboard: list[dict[str, Any]] = []
    for (model, alpha, retrieve_top_k, rerank_top_n), rows in grouped.items():
        quality = mean(row.quality_score for row in rows)
        cost = mean(row.cost_usd for row in rows)
        latency = mean(row.latency_ms for row in rows)
        failure_rate = sum(1 for row in rows if row.failure_category != "passed") / max(1, len(rows))
        metrics = [row.metrics_json for row in rows]
        config_leaderboard.append(
            {
                "model": model,
                "alpha": alpha,
                "retrieve_top_k": retrieve_top_k,
                "rerank_top_n": rerank_top_n,
                "quality_score": round(quality, 4),
                "cost_per_query": round(cost, 6),
                "avg_latency_ms": round(latency),
                "failure_rate": round(failure_rate, 4),
                "answer_correctness": round(mean(m["answer_correctness"] for m in metrics), 4),
                "groundedness": round(mean(m["groundedness"] for m in metrics), 4),
                "citation_accuracy": round(mean(m["citation_accuracy"] for m in metrics), 4),
                "refusal_correctness": round(mean(m["refusal_correctness"] for m in metrics), 4),
            }
        )

    config_leaderboard.sort(
        key=lambda item: (item["quality_score"], -item["cost_per_query"]), reverse=True
    )
    best_config = config_leaderboard[0] if config_leaderboard else None
    failure_breakdown = Counter(result.failure_category for result in results)
    real_failures = Counter(
        result.failure_category for result in results if result.failure_category != "passed"
    )

    model_summary: dict[str, dict[str, Any]] = {}
    model_leaderboard: list[dict[str, Any]] = []
    for model in {result.model for result in results}:
        rows = [result for result in results if result.model == model]
        quality_score = round(mean(row.quality_score for row in rows), 4)
        cost_per_query = round(mean(row.cost_usd for row in rows), 6)
        avg_latency_ms = round(mean(row.latency_ms for row in rows))
        failure_rate = round(
            sum(1 for row in rows if row.failure_category != "passed") / max(1, len(rows)),
            4,
        )
        model_configs = [item for item in config_leaderboard if item["model"] == model]
        model_best_config = model_configs[0] if model_configs else None
        row = {
            "model": model,
            "quality_score": quality_score,
            "cost_per_query": cost_per_query,
            "avg_latency_ms": avg_latency_ms,
            "failure_rate": failure_rate,
            "best_alpha": model_best_config["alpha"] if model_best_config else None,
            "best_retrieve_top_k": model_best_config["retrieve_top_k"] if model_best_config else None,
            "best_rerank_top_n": model_best_config["rerank_top_n"] if model_best_config else None,
        }
        model_summary[model] = {
            "quality_score": round(mean(row.quality_score for row in rows), 4),
            "cost_per_query": round(mean(row.cost_usd for row in rows), 6),
            "avg_latency_ms": round(mean(row.latency_ms for row in rows)),
            "failure_rate": failure_rate,
        }
        model_leaderboard.append(row)

    model_leaderboard.sort(
        key=lambda item: (item["quality_score"], -item["cost_per_query"]), reverse=True
    )
    quality_cost_points = [
        {
            **item,
            "label": item["model"],
            "config_label": f"a={item['alpha']} k={item['retrieve_top_k']} n={item['rerank_top_n']}",
        }
        for item in config_leaderboard
    ]
    failure_by_model_counter = Counter(
        (result.model, result.failure_category)
        for result in results
        if result.failure_category != "passed"
    )
    failure_by_model = [
        {"model": model, "category": category, "count": count}
        for (model, category), count in failure_by_model_counter.items()
    ]
    most_common_failure = None
    if real_failures:
        category, count = real_failures.most_common(1)[0]
        most_common_failure = {"category": category, "count": count}
    acceptable = [item for item in model_leaderboard if item["quality_score"] >= 0.8]
    cheapest = min(acceptable, key=lambda item: item["cost_per_query"], default=None)
    fastest = min(acceptable, key=lambda item: item["avg_latency_ms"], default=None)

    return {
        "best_config": best_config,
        "best_model": model_leaderboard[0] if model_leaderboard else None,
        "cheapest_acceptable_model": cheapest,
        "fastest_acceptable_model": fastest,
        "most_common_failure": most_common_failure,
        "leaderboard": config_leaderboard,
        "config_leaderboard": config_leaderboard,
        "model_leaderboard": model_leaderboard,
        "quality_cost_points": quality_cost_points,
        "failure_breakdown": [
            {"category": category, "count": count} for category, count in failure_breakdown.items()
        ],
        "failure_by_model": failure_by_model,
        "model_summary": model_summary,
        "result_count": len(results),
    }
