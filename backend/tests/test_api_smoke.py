from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

TEST_DB_PATH = Path(tempfile.gettempdir()) / f"supportbench_api_smoke_{os.getpid()}.db"
if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
os.environ["USE_PINECONE"] = "false"
os.environ["USE_REAL_MODELS"] = "false"

from app.config import settings

settings.database_url = f"sqlite:///{TEST_DB_PATH}"
settings.use_pinecone = False
settings.use_real_models = False
settings.eval_judge_provider = "deterministic"


class ApiSmokeTests(unittest.TestCase):
    def test_ingest_ask_eval_dashboard(self) -> None:
        settings.use_pinecone = False
        settings.use_real_models = False
        settings.eval_judge_provider = "deterministic"

        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as client:
            ingest = client.post("/documents/ingest")
            self.assertEqual(ingest.status_code, 200)
            self.assertGreaterEqual(ingest.json()["documents"], 5)

            empty_dashboard = client.get("/dashboard/summary")
            self.assertEqual(empty_dashboard.status_code, 200)
            self.assertEqual(empty_dashboard.json()["dashboard_state"], "empty")
            self.assertFalse(empty_dashboard.json()["config_leaderboard"])

            answer = client.post(
                "/ask",
                json={
                    "question": "What is the Pro API rate limit?",
                    "model": "openai_primary",
                    "alpha": 0.5,
                    "retrieve_top_k": 10,
                    "rerank_top_n": 3,
                },
            )
            self.assertEqual(answer.status_code, 200)
            self.assertIn("answer", answer.json())
            self.assertIn("trace_id", answer.json())

            run = client.post(
                "/eval/run",
                json={
                    "run_name": "api-smoke",
                    "question_limit": 2,
                    "models": ["groq_llama_3_1_8b", "openai_primary"],
                    "alphas": [0.5],
                    "retrieve_top_k": [10],
                    "rerank_top_n": [3],
                },
            )
            self.assertEqual(run.status_code, 200)
            run_payload = run.json()
            self.assertEqual(run_payload["total_results"], 4)
            self.assertEqual(run_payload["summary"]["dashboard_state"], "completed")
            self.assertEqual(run_payload["summary"]["run_id"], run_payload["run_id"])
            self.assertEqual(
                run_payload["summary"]["best_config"],
                run_payload["summary"]["config_leaderboard"][0],
            )

            dashboard = client.get("/dashboard/summary")
            self.assertEqual(dashboard.status_code, 200)
            payload = dashboard.json()
            self.assertEqual(payload["dashboard_state"], "completed")
            self.assertTrue(payload["model_leaderboard"])
            self.assertTrue(payload["config_leaderboard"])

            reingest = client.post("/documents/ingest")
            self.assertEqual(reingest.status_code, 200)
            stale_dashboard = client.get("/dashboard/summary")
            self.assertEqual(stale_dashboard.status_code, 200)
            stale_payload = stale_dashboard.json()
            self.assertEqual(stale_payload["dashboard_state"], "empty")
            self.assertEqual(stale_payload["run_id"], run_payload["run_id"])
            self.assertTrue(stale_payload["config_leaderboard"])
            self.assertTrue(stale_payload["is_stale"])
            self.assertIn("re-ingested", stale_payload["summary_notice"])

            run_details = client.get(f"/eval/runs/{run_payload['run_id']}")
            self.assertEqual(run_details.status_code, 200)
            first_result = run_details.json()["results"][0]
            self.assertIn("question", first_result)
            self.assertIn("reference_answer", first_result)
            self.assertIn("expected_doc", first_result)
            self.assertIn("expected_section", first_result)
            self.assertIn("expected_sources", first_result)
            self.assertIn("question_type", first_result)
            self.assertIn("source_recall", first_result)
            self.assertIn("citation_matched", first_result)
            self.assertIn("top_retrieved_doc", first_result)
            self.assertIn("top_retrieved_section", first_result)
            self.assertIn("expected_source_found", first_result)

            export = client.get(f"/eval/runs/{run_payload['run_id']}/export.csv")
            self.assertEqual(export.status_code, 200)
            self.assertEqual(export.headers["content-type"].split(";")[0], "text/csv")
            export_text = export.text
            self.assertIn("reference_answer", export_text)
            self.assertIn("generated_answer", export_text)
            self.assertIn("expected_source_found", export_text)
            self.assertIn("question_type", export_text)
            self.assertIn("source_recall", export_text)
            self.assertIn("judge_source", export_text)

            missing_export = client.get("/eval/runs/999999/export.csv")
            self.assertEqual(missing_export.status_code, 404)

            from app.db import SessionLocal
            from app.db_models import ExperimentResultRecord, ExperimentRunRecord

            with SessionLocal() as db:
                retired_run = ExperimentRunRecord(
                    run_name="retired-provider",
                    status="completed",
                    completed_at=datetime.utcnow(),
                    config_grid_json={"models": ["retired_model"]},
                    summary_json={
                        "best_config": {"model": "retired_model"},
                        "result_count": 1,
                    },
                )
                db.add(retired_run)
                db.commit()
                db.refresh(retired_run)
                db.add(
                    ExperimentResultRecord(
                        run_id=retired_run.id,
                        question_id="retired-q1",
                        model="retired_model",
                        alpha=0.5,
                        retrieve_top_k=10,
                        rerank_top_n=3,
                        quality_score=1.0,
                        cost_usd=0.0,
                        latency_ms=1,
                        failure_category="passed",
                        answer_json={},
                        retrieved_json=[],
                        metrics_json={
                            "answer_correctness": 1,
                            "groundedness": 1,
                            "citation_accuracy": 1,
                            "refusal_correctness": 1,
                        },
                        trace_id="retired-trace",
                    )
                )
                db.commit()

            history = client.get("/dashboard/history")
            self.assertEqual(history.status_code, 200)
            history_payload = history.json()
            self.assertTrue(history_payload["model_performance"])
            self.assertTrue(history_payload["config_performance"])
            history_models = {item["model"] for item in history_payload["model_performance"]}
            self.assertNotIn("retired_model", history_models)

            models = client.get("/models")
            self.assertEqual(models.status_code, 200)
            aliases = {item["alias"] for item in models.json()["models"]}
            self.assertEqual(
                aliases,
                {
                    "groq_llama_3_1_8b",
                    "groq_llama_3_3_70b",
                    "groq_gpt_oss_20b",
                    "openai_primary",
                    "gemini_flash",
                },
            )

    def test_dashboard_summary_fields_are_consistent(self) -> None:
        os.environ["USE_PINECONE"] = "false"
        os.environ["USE_REAL_MODELS"] = "false"

        from app.services.repository import summarize_results

        results = [
            _result("model_a", 0.25, 10, 5, 0.95, 0.020, 500),
            _result("model_a", 0.25, 10, 5, 0.85, 0.020, 500),
            _result("model_a", 0.5, 10, 5, 0.70, 0.010, 400, "wrong_answer"),
            _result("model_b", 0.25, 10, 3, 0.83, 0.005, 300),
            _result("model_c", 0.75, 20, 5, 0.81, 0.003, 900),
        ]
        summary = summarize_results(results)

        self.assertEqual(summary["best_config"], summary["config_leaderboard"][0])
        self.assertEqual(summary["best_model"], summary["model_leaderboard"][0])
        self.assertEqual(summary["best_config"]["model"], "model_a")
        self.assertEqual(summary["best_config"]["alpha"], 0.25)
        self.assertEqual(summary["best_config"]["retrieve_top_k"], 10)
        self.assertEqual(summary["best_config"]["rerank_top_n"], 5)

        model_a = next(row for row in summary["model_leaderboard"] if row["model"] == "model_a")
        self.assertEqual(model_a["best_alpha"], 0.25)
        self.assertEqual(model_a["best_retrieve_top_k"], 10)
        self.assertEqual(model_a["best_rerank_top_n"], 5)

        acceptable_models = [
            row for row in summary["model_leaderboard"] if row["quality_score"] >= 0.8
        ]
        cheapest = min(acceptable_models, key=lambda item: item["cost_per_query"])
        fastest = min(acceptable_models, key=lambda item: item["avg_latency_ms"])
        self.assertEqual(summary["cheapest_acceptable_model"], cheapest)
        self.assertEqual(summary["fastest_acceptable_model"], fastest)

        for row in summary["quality_cost_points"]:
            matching_config = next(
                config
                for config in summary["config_leaderboard"]
                if (
                    config["model"] == row["model"]
                    and config["alpha"] == row["alpha"]
                    and config["retrieve_top_k"] == row["retrieve_top_k"]
                    and config["rerank_top_n"] == row["rerank_top_n"]
                )
            )
            self.assertEqual(row["quality_score"], matching_config["quality_score"])
            self.assertEqual(row["cost_per_query"], matching_config["cost_per_query"])


def _result(
    model: str,
    alpha: float,
    retrieve_top_k: int,
    rerank_top_n: int,
    quality_score: float,
    cost_usd: float,
    latency_ms: int,
    failure_category: str = "passed",
):
    from app.db_models import ExperimentResultRecord

    return ExperimentResultRecord(
        run_id=1,
        question_id=f"q-{model}-{alpha}-{retrieve_top_k}-{rerank_top_n}-{quality_score}",
        model=model,
        alpha=alpha,
        retrieve_top_k=retrieve_top_k,
        rerank_top_n=rerank_top_n,
        quality_score=quality_score,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        failure_category=failure_category,
        answer_json={"estimated_cost_usd": cost_usd, "latency_ms": latency_ms},
        retrieved_json=[],
        metrics_json={
            "answer_correctness": quality_score,
            "groundedness": quality_score,
            "citation_accuracy": 1.0 if failure_category == "passed" else 0.5,
            "refusal_correctness": 1.0,
        },
        trace_id=f"trace-{model}-{quality_score}",
    )


if __name__ == "__main__":
    unittest.main()
