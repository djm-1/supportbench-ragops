from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
import unittest

from app.core.bm25 import BM25Index
from app.core.chunker import chunk_sections
from app.core.evaluator import evaluate_answer
from app.core.fusion import fuse_candidates
from app.core.judge import deterministic_judge
from app.core.parser import parse_support_docs
from app.core.reranker import LexicalReranker
from app.core.retriever import HybridRetriever
from app.core.types import AnswerJudgeResult, AnswerResult, EvalQuestion, RetrievalCandidate, SupportChunk


class CoreRagTests(unittest.TestCase):
    def setUp(self) -> None:
        from app.config import settings

        settings.use_pinecone = False
        settings.use_real_models = False
        settings.eval_judge_provider = "deterministic"

    def test_chunker_preserves_metadata(self) -> None:
        sections = [
            {
                "document": "billing.md",
                "title": "Billing",
                "section": "Refund Policy",
                "text": "Annual plans are refundable within 14 days after purchase.",
            }
        ]
        chunks = chunk_sections(sections, max_words=12, overlap_words=2)
        self.assertEqual(chunks[0].document, "billing.md")
        self.assertEqual(chunks[0].section, "Refund Policy")
        self.assertEqual(chunks[0].metadata["title"], "Billing")
        self.assertTrue(chunks[0].chunk_id.startswith("chunk_"))

    def test_bm25_ranks_matching_document_first(self) -> None:
        index = BM25Index(
            [
                "Free workspaces have 60 API requests per minute.",
                "Password reset links expire after 30 minutes.",
                "Slack notifications can be configured per project.",
            ]
        )
        results = index.search("API request limit", limit=2)
        self.assertEqual(results[0][0], 0)
        self.assertGreater(results[0][1], results[1][1])

    def test_alpha_fusion_prefers_dense_when_alpha_high(self) -> None:
        chunks = [
            SupportChunk("a", "a.md", "A", "dense winner"),
            SupportChunk("b", "b.md", "B", "sparse winner"),
        ]
        candidates = [
            RetrievalCandidate(chunks[0], dense_score=0.9, sparse_score=0.1),
            RetrievalCandidate(chunks[1], dense_score=0.1, sparse_score=0.9),
        ]
        dense_first = fuse_candidates(candidates, alpha=0.75)
        self.assertEqual(dense_first[0].chunk.chunk_id, "a")

    def test_reranker_respects_limit(self) -> None:
        candidates = [
            RetrievalCandidate(SupportChunk("a", "a.md", "Rate Limits", "API rate limit"), 1, 1, 1),
            RetrievalCandidate(SupportChunk("b", "b.md", "Refunds", "billing refund"), 1, 1, 1),
        ]
        reranked = LexicalReranker().rerank("What is the API rate limit?", candidates, limit=1)
        self.assertEqual(len(reranked), 1)
        self.assertEqual(reranked[0].chunk.chunk_id, "a")

    def test_evaluator_scores_supported_answer(self) -> None:
        question = EvalQuestion(
            id="q",
            question="How long are password reset links valid?",
            reference_answer="Password reset links expire after 30 minutes and can be used only once.",
            expected_doc="security.md",
            expected_section="Password Reset",
            tags=["security"],
        )
        answer = AnswerResult(
            answer="Password reset links expire after 30 minutes and can be used only once.",
            citations=[
                {"document": "security.md", "section": "Password Reset", "chunk_id": "chunk_1"}
            ],
            model="demo",
            input_tokens=20,
            output_tokens=12,
            estimated_cost_usd=0.001,
            latency_ms=100,
        )
        candidates = [
            RetrievalCandidate(
                SupportChunk(
                    "chunk_1",
                    "security.md",
                    "Password Reset",
                    "Password reset links expire after 30 minutes and can be used only once.",
                )
            )
        ]
        metrics = evaluate_answer(question, answer, candidates)
        self.assertEqual(metrics.failure_category, "passed")
        self.assertEqual(metrics.citation_accuracy, 1.0)
        self.assertGreater(metrics.quality_score, 0.9)

    def test_evaluator_catches_unsupported_citation(self) -> None:
        question = EvalQuestion(
            id="q",
            question="What is the Pro API limit?",
            reference_answer="Pro workspaces are limited to 600 API requests per minute.",
            expected_doc="api_limits.md",
            expected_section="Rate Limits",
            tags=["api"],
        )
        answer = AnswerResult(
            answer="Pro workspaces are limited to 600 API requests per minute.",
            citations=[{"document": "billing.md", "section": "Refunds", "chunk_id": "wrong"}],
            model="demo",
            input_tokens=20,
            output_tokens=12,
            estimated_cost_usd=0.001,
            latency_ms=100,
        )
        candidates = [
            RetrievalCandidate(
                SupportChunk(
                    "chunk_1",
                    "api_limits.md",
                    "Rate Limits",
                    "Pro workspaces are limited to 600 API requests per minute.",
                )
            )
        ]
        metrics = evaluate_answer(question, answer, candidates)
        self.assertEqual(metrics.failure_category, "wrong_citation")
        self.assertEqual(metrics.citation_matched, False)
        self.assertEqual(metrics.source_recall, 1.0)

    def test_evaluator_scores_multi_source_retrieval_and_citation(self) -> None:
        question = EvalQuestion(
            id="q",
            question="How does membership change shipping for a small order?",
            reference_answer="Members get free shipping on all orders, while non-members below Rs 499 pay Rs 50.",
            expected_doc="membership.md",
            expected_section="Membership",
            expected_sources=[
                {"document": "membership.md", "section": "Membership"},
                {"document": "shipping.md", "section": "Shipping Fees"},
            ],
            tags=["cross_document"],
            question_type="cross_document",
        )
        answer = AnswerResult(
            answer="Members get free shipping on all orders. Non-members below Rs 499 pay Rs 50.",
            citations=[{"document": "membership.md", "section": "Membership", "chunk_id": "chunk_1"}],
            model="demo",
            input_tokens=20,
            output_tokens=12,
            estimated_cost_usd=0.001,
            latency_ms=100,
        )
        candidates = [
            RetrievalCandidate(SupportChunk("chunk_1", "membership.md", "Membership", "Members get free shipping.")),
            RetrievalCandidate(SupportChunk("chunk_2", "shipping.md", "Shipping Fees", "Non-members below Rs 499 pay Rs 50.")),
        ]
        metrics = evaluate_answer(
            question,
            answer,
            candidates,
            AnswerJudgeResult(score=0.95, label="correct", rationale="All facts match."),
        )
        self.assertEqual(metrics.source_recall, 1.0)
        self.assertEqual(metrics.retrieval_hit, 1.0)
        self.assertEqual(metrics.citation_accuracy, 0.5)
        self.assertEqual(metrics.failure_category, "partial_citation")
        self.assertIn("citation", metrics.deterministic_gate_failures)

    def test_evaluator_accepts_paraphrase_with_judge(self) -> None:
        question = EvalQuestion(
            id="q",
            question="How long are password reset links valid?",
            reference_answer="Password reset links expire after 30 minutes and can be used only once.",
            expected_doc="security.md",
            expected_section="Password Reset",
            tags=["security"],
        )
        answer = AnswerResult(
            answer="The reset link works one time and expires in 30 minutes.",
            citations=[
                {"document": "security.md", "section": "Password Reset", "chunk_id": "chunk_1"}
            ],
            model="demo",
            input_tokens=20,
            output_tokens=12,
            estimated_cost_usd=0.001,
            latency_ms=100,
        )
        candidates = [
            RetrievalCandidate(
                SupportChunk(
                    "chunk_1",
                    "security.md",
                    "Password Reset",
                    "Password reset links expire after 30 minutes and can be used only once.",
                )
            )
        ]
        metrics = evaluate_answer(
            question,
            answer,
            candidates,
            AnswerJudgeResult(
                score=0.94,
                label="correct",
                rationale="Same facts with different wording.",
            ),
        )
        self.assertEqual(metrics.failure_category, "passed")
        self.assertEqual(metrics.judge_label, "correct")

    def test_evaluator_flags_missing_key_number(self) -> None:
        question = EvalQuestion(
            id="q",
            question="How long are gift cards valid?",
            reference_answer="All Gift cards are valid till October 2026.",
            expected_doc="gift.md",
            expected_section="Gift Validity",
            tags=["gift"],
        )
        answer = AnswerResult(
            answer="Gift cards are valid for a while.",
            citations=[{"document": "gift.md", "section": "Gift Validity", "chunk_id": "chunk_1"}],
            model="demo",
            input_tokens=20,
            output_tokens=12,
            estimated_cost_usd=0.001,
            latency_ms=100,
        )
        candidates = [
            RetrievalCandidate(
                SupportChunk("chunk_1", "gift.md", "Gift Validity", "All Gift cards are valid till October 2026.")
            )
        ]
        metrics = evaluate_answer(question, answer, candidates)
        self.assertEqual(metrics.failure_category, "wrong_answer")
        self.assertIn("2026", metrics.missing_facts)

    def test_correct_refusal_is_right_and_not_ungrounded(self) -> None:
        question = EvalQuestion(
            id="q",
            question="Can I return an opened phone cover after using it for a week?",
            reference_answer=(
                "The docs do not give a specific phone-cover-used-for-a-week return rule. "
                "They only state general return eligibility and that used, worn, damaged, "
                "or missing-tag items may be refused."
            ),
            expected_doc=None,
            expected_section=None,
            tags=["missing_data"],
            should_refuse=True,
            question_type="missing_data",
        )
        answer = AnswerResult(
            answer="The provided support docs do not state enough information to answer this question.",
            citations=[],
            model="demo",
            input_tokens=20,
            output_tokens=12,
            estimated_cost_usd=0.001,
            latency_ms=100,
        )
        candidates = [
            RetrievalCandidate(
                SupportChunk(
                    "chunk_1",
                    "returns.md",
                    "Return Policy",
                    "Customers can return orders subject to policy and eligibility.",
                )
            )
        ]

        metrics = evaluate_answer(question, answer, candidates)

        self.assertEqual(metrics.failure_category, "passed")
        self.assertEqual(metrics.result_label, "right")
        self.assertEqual(metrics.issue_label, "none")
        self.assertEqual(metrics.groundedness, 1.0)
        self.assertNotIn("grounding", metrics.deterministic_gate_failures)

    def test_unsupported_question_with_hallucinated_answer_is_wrong(self) -> None:
        question = EvalQuestion(
            id="q",
            question="Can I return an opened phone cover after using it for a week?",
            reference_answer="The docs do not give a specific phone-cover-used-for-a-week return rule.",
            expected_doc=None,
            expected_section=None,
            tags=["missing_data"],
            should_refuse=True,
            question_type="missing_data",
        )
        answer = AnswerResult(
            answer="Yes, opened phone covers can always be returned after one week.",
            citations=[],
            model="demo",
            input_tokens=20,
            output_tokens=12,
            estimated_cost_usd=0.001,
            latency_ms=100,
        )
        metrics = evaluate_answer(question, answer, [])

        self.assertEqual(metrics.failure_category, "should_have_refused")
        self.assertEqual(metrics.result_label, "wrong")
        self.assertEqual(metrics.issue_label, "unsupported_answer")

    def test_rule_based_fallback_marks_concise_key_fact_as_partial(self) -> None:
        question = EvalQuestion(
            id="q",
            question="What is the return window after delivery?",
            reference_answer="Customers can return an order within 30 days after delivery, subject to the return policy and product eligibility.",
            expected_doc="returns.md",
            expected_section="Return Policy",
            tags=["returns"],
            reference_facts=[
                "30 days after delivery",
                "subject to return policy and eligibility",
            ],
        )
        answer = AnswerResult(
            answer="30 days",
            citations=[{"document": "returns.md", "section": "Return Policy", "chunk_id": "chunk_1"}],
            model="demo",
            input_tokens=20,
            output_tokens=2,
            estimated_cost_usd=0.001,
            latency_ms=100,
        )
        candidates = [
            RetrievalCandidate(
                SupportChunk(
                    "chunk_1",
                    "returns.md",
                    "Return Policy",
                    question.reference_answer,
                )
            )
        ]

        judge_result = deterministic_judge(question, answer)
        metrics = evaluate_answer(question, answer, candidates, judge_result)

        self.assertEqual(judge_result.label, "partial")
        self.assertEqual(metrics.failure_category, "partial_answer")
        self.assertEqual(metrics.result_label, "partial")
        self.assertEqual(metrics.issue_label, "incomplete_answer")
        self.assertNotEqual(metrics.failure_category, "wrong_answer")
        self.assertIn("subject to return policy and eligibility", metrics.missing_facts)

    def test_rule_based_fallback_keeps_wrong_key_number_as_wrong_answer(self) -> None:
        question = EvalQuestion(
            id="q",
            question="What is the return window after delivery?",
            reference_answer="Customers can return an order within 30 days after delivery, subject to the return policy and product eligibility.",
            expected_doc="returns.md",
            expected_section="Return Policy",
            tags=["returns"],
            reference_facts=["30 days after delivery"],
        )
        answer = AnswerResult(
            answer="15 days",
            citations=[{"document": "returns.md", "section": "Return Policy", "chunk_id": "chunk_1"}],
            model="demo",
            input_tokens=20,
            output_tokens=2,
            estimated_cost_usd=0.001,
            latency_ms=100,
        )
        candidates = [
            RetrievalCandidate(
                SupportChunk("chunk_1", "returns.md", "Return Policy", question.reference_answer)
            )
        ]

        metrics = evaluate_answer(question, answer, candidates, deterministic_judge(question, answer))

        self.assertEqual(metrics.failure_category, "wrong_answer")
        self.assertTrue(metrics.contradictions)

    def test_openai_judge_uses_responses_api_for_nano_model(self) -> None:
        from app.core.judge import _uses_responses_api

        self.assertTrue(_uses_responses_api("gpt-5.4-nano"))
        self.assertFalse(_uses_responses_api("gpt-4o-mini"))

    def test_openai_responses_text_and_cost_helpers(self) -> None:
        from app.core.judge import _extract_responses_text, _judge_from_json

        response = {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"score":0.9,"label":"correct","rationale":"same facts","missing_facts":[],"contradictions":[]}',
                        }
                    ]
                }
            ]
        }
        self.assertIn('"score":0.9', _extract_responses_text(response))

        result = _judge_from_json(
            _extract_responses_text(response),
            judge_model="openai:gpt-5.4-nano",
            latency_ms=12,
            input_tokens=1000,
            output_tokens=100,
        )
        self.assertEqual(result.label, "correct")
        self.assertEqual(result.judge_cost_usd, 0.000325)

    def test_judge_failure_marks_unavailable_without_provider_fallback(self) -> None:
        from app.config import settings
        import app.core.judge as judge_module

        question = EvalQuestion(
            id="q",
            question="What denominations are gift cards available in?",
            reference_answer="Gift cards are currently available in denominations of INR 2000.",
            expected_doc="gift.md",
            expected_section="Gift Cards",
            tags=["gift"],
            question_type="paraphrase",
        )
        answer = AnswerResult(
            answer="Gift cards are available in INR 2000.",
            citations=[{"document": "gift.md", "section": "Gift Cards", "chunk_id": "chunk_1"}],
            model="demo",
            input_tokens=20,
            output_tokens=12,
            estimated_cost_usd=0.0,
            latency_ms=1,
        )
        candidates = [
            RetrievalCandidate(
                SupportChunk("chunk_1", "gift.md", "Gift Cards", question.reference_answer)
            )
        ]
        original_responses_judge = judge_module._openai_responses_judge
        settings.use_real_models = True
        settings.eval_judge_provider = "openai"
        settings.eval_judge_model = "gpt-5.4-nano"
        settings.openai_api_key = "test-key"
        try:
            def fail_judge(*_args, **_kwargs):
                raise RuntimeError("model unavailable")

            judge_module._openai_responses_judge = fail_judge
            result = judge_module.judge_answer(question, answer, candidates)
        finally:
            judge_module._openai_responses_judge = original_responses_judge

        self.assertEqual(result.label, "judge_unavailable")
        self.assertEqual(result.judge_model, "openai:gpt-5.4-nano")
        self.assertTrue(result.judge_unavailable)

    def test_concise_direct_answer_escalates_to_llm_judge_when_available(self) -> None:
        from app.config import settings
        import app.core.judge as judge_module

        question = EvalQuestion(
            id="q",
            question="What is the return window after delivery?",
            reference_answer="Customers can return an order within 30 days after delivery, subject to the return policy and product eligibility.",
            expected_doc="returns.md",
            expected_section="Return Policy",
            tags=["returns"],
            reference_facts=[
                "30 days after delivery",
                "subject to return policy and eligibility",
            ],
        )
        answer = AnswerResult(
            answer="30 days",
            citations=[{"document": "returns.md", "section": "Return Policy", "chunk_id": "chunk_1"}],
            model="demo",
            input_tokens=20,
            output_tokens=2,
            estimated_cost_usd=0.0,
            latency_ms=1,
        )
        candidates = [
            RetrievalCandidate(SupportChunk("chunk_1", "returns.md", "Return Policy", question.reference_answer))
        ]
        original_responses_judge = judge_module._openai_responses_judge
        settings.use_real_models = True
        settings.eval_judge_provider = "openai"
        settings.eval_judge_model = "gpt-5.4-nano"
        settings.openai_api_key = "test-key"
        try:
            def fake_judge(*_args, **_kwargs):
                return AnswerJudgeResult(
                    score=0.68,
                    label="partial",
                    rationale="Correct return window, but missing caveats.",
                    missing_facts=["subject to return policy and eligibility"],
                    judge_model="openai:gpt-5.4-nano",
                    judge_source="llm",
                )

            judge_module._openai_responses_judge = fake_judge
            result = judge_module.judge_answer(question, answer, candidates)
        finally:
            judge_module._openai_responses_judge = original_responses_judge

        self.assertEqual(result.judge_source, "llm")
        self.assertEqual(result.label, "partial")

    def test_generated_souled_store_dataset_shape(self) -> None:
        dataset_dir = Path(__file__).resolve().parents[2] / "data" / "generated" / "souled_store"
        if not dataset_dir.exists():
            self.skipTest("Generated Souled Store dataset is local-only and not present.")
        manifest = json.loads((dataset_dir / "source_manifest.json").read_text(encoding="utf-8"))
        questions = json.loads((dataset_dir / "eval_questions.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["faq_pair_count"], 59)
        self.assertEqual(len(questions), 50)
        self.assertEqual(
            Counter(question["question_type"] for question in questions),
            Counter(
                {
                    "direct": 10,
                    "paraphrase": 10,
                    "calculation": 10,
                    "missing_data": 10,
                    "cross_document": 10,
                }
            ),
        )
        sections = parse_support_docs(dataset_dir)
        expected_sources = set()
        for question in questions:
            for source in question.get("expected_sources", []):
                expected_sources.add((source["document"], source["section"]))
        available_sources = {(section["document"], section["section"]) for section in sections}
        self.assertTrue(expected_sources <= available_sources)

    def test_generated_souled_store_retrieves_expected_source(self) -> None:
        dataset_dir = Path(__file__).resolve().parents[2] / "data" / "generated" / "souled_store"
        if not dataset_dir.exists():
            self.skipTest("Generated Souled Store dataset is local-only and not present.")
        questions = json.loads((dataset_dir / "eval_questions.json").read_text(encoding="utf-8"))
        question = next(item for item in questions if item["id"] == "tss_para_009")
        chunks = chunk_sections(parse_support_docs(dataset_dir), max_words=900, overlap_words=0)
        results = HybridRetriever(chunks).retrieve(
            question["question"],
            alpha=0.5,
            retrieve_top_k=10,
            rerank_top_n=5,
        )
        self.assertTrue(
            any(
                candidate.chunk.document == question["expected_doc"]
                and candidate.chunk.section == question["expected_section"]
                for candidate in results
            )
        )


if __name__ == "__main__":
    unittest.main()
