"""
test_eval_harness.py
Unit tests for the retrieval eval harness (pure Python, no LLM calls).

These tests cover:
  - recall@k calculation correctness
  - MRR (reciprocal rank) correctness
  - summarize_retrieval_results aggregation
  - chunk_text sentence-aware chunker behavior

No network calls are made in these tests.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.eval_harness import (
    recall_at_k,
    reciprocal_rank,
    summarize_retrieval_results,
)
from src.models import EvalQAPair, RetrievalEvalResult


# ── recall_at_k ────────────────────────────────────────────────────────────────

class TestRecallAtK:

    def test_perfect_recall_single_gt(self):
        """Single ground truth doc found at rank 1."""
        assert recall_at_k(["PM-001"], ["PM-001", "CB-002", "CR-003"], k=5) == 1.0

    def test_zero_recall_doc_not_retrieved(self):
        """Ground truth doc not in retrieved list."""
        assert recall_at_k(["PM-999"], ["CB-001", "CB-002", "CB-003"], k=5) == 0.0

    def test_partial_recall_multiple_gt(self):
        """2 of 4 ground truth docs retrieved in top-5."""
        result = recall_at_k(
            ground_truth_ids=["PM-001", "CB-001", "CR-001", "WL-001"],
            retrieved_ids=["PM-001", "CB-001", "ST-001", "ST-002", "ST-003"],
            k=5,
        )
        assert result == pytest.approx(0.5)

    def test_recall_at_1_vs_5(self):
        """recall@1 should be ≤ recall@5 for the same query."""
        gt = ["PM-001", "PM-002"]
        retrieved = ["CB-001", "PM-001", "PM-002", "CB-002", "CB-003"]
        r1 = recall_at_k(gt, retrieved, k=1)
        r5 = recall_at_k(gt, retrieved, k=5)
        assert r1 <= r5

    def test_empty_ground_truth_returns_zero(self):
        """Empty ground truth list should not raise, returns 0."""
        assert recall_at_k([], ["PM-001", "CB-001"], k=5) == 0.0

    def test_k_larger_than_retrieved_list(self):
        """k can safely exceed the length of the retrieved list."""
        assert recall_at_k(["PM-001"], ["PM-001"], k=20) == 1.0

    def test_gt_doc_at_boundary_of_k(self):
        """GT doc at exactly rank k should be counted."""
        retrieved = ["CB-001", "CB-002", "CB-003", "CB-004", "PM-001"]
        assert recall_at_k(["PM-001"], retrieved, k=5) == 1.0
        # But not if k=4
        assert recall_at_k(["PM-001"], retrieved, k=4) == 0.0


# ── reciprocal_rank ────────────────────────────────────────────────────────────

class TestReciprocalRank:

    def test_first_hit_at_rank_1(self):
        assert reciprocal_rank(["PM-001"], ["PM-001", "CB-001"]) == pytest.approx(1.0)

    def test_first_hit_at_rank_2(self):
        assert reciprocal_rank(["PM-001"], ["CB-001", "PM-001"]) == pytest.approx(0.5)

    def test_first_hit_at_rank_5(self):
        retrieved = ["CB-001", "CB-002", "CB-003", "CB-004", "PM-001"]
        assert reciprocal_rank(["PM-001"], retrieved) == pytest.approx(0.2)

    def test_no_hit_returns_zero(self):
        assert reciprocal_rank(["PM-999"], ["CB-001", "CB-002"]) == 0.0

    def test_multiple_gt_uses_first_hit(self):
        """When multiple GT docs exist, RR is based on the FIRST one found."""
        retrieved = ["CB-001", "PM-002", "CB-002", "PM-001"]
        rr = reciprocal_rank(["PM-001", "PM-002"], retrieved)
        # PM-002 is at rank 2 → RR = 0.5
        assert rr == pytest.approx(0.5)


# ── summarize_retrieval_results ────────────────────────────────────────────────

class TestSummarizeRetrieval:

    def _make_result(self, r1, r3, r5, r10, rr) -> RetrievalEvalResult:
        return RetrievalEvalResult(
            qa_id="qa-test",
            question="test",
            ground_truth_doc_ids=["PM-001"],
            retrieved_doc_ids=["PM-001"],
            recall_at_1=r1,
            recall_at_3=r3,
            recall_at_5=r5,
            recall_at_10=r10,
            reciprocal_rank=rr,
        )

    def test_empty_results(self):
        summary = summarize_retrieval_results([])
        assert "error" in summary

    def test_single_perfect_result(self):
        results = [self._make_result(1.0, 1.0, 1.0, 1.0, 1.0)]
        summary = summarize_retrieval_results(results)
        assert summary["recall_at_5"] == 1.0
        assert summary["mrr"] == 1.0
        assert summary["n_queries"] == 1

    def test_average_across_two_results(self):
        results = [
            self._make_result(1.0, 1.0, 1.0, 1.0, 1.0),
            self._make_result(0.0, 0.0, 0.0, 0.0, 0.0),
        ]
        summary = summarize_retrieval_results(results)
        assert summary["recall_at_5"] == pytest.approx(0.5)
        assert summary["mrr"] == pytest.approx(0.5)

    def test_mrr_field_present(self):
        results = [self._make_result(0.5, 0.5, 0.75, 0.75, 0.333)]
        summary = summarize_retrieval_results(results)
        assert "mrr" in summary
        assert "recall_at_1" in summary
        assert "recall_at_10" in summary


# ── chunk_text ─────────────────────────────────────────────────────────────────

class TestChunkText:
    """Test the sentence-aware chunker from ingest.py."""

    def test_short_text_is_single_chunk(self):
        from scripts.ingest import chunk_text
        text = "This is a short sentence. It should produce one chunk."
        chunks = chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        from scripts.ingest import chunk_text
        # ~200 words repeated → should exceed 512 tokens and produce 2+ chunks
        sentence = "The campaign delivered strong results across all channels and exceeded the target ROAS. "
        text = sentence * 60
        chunks = chunk_text(text, chunk_size=128, overlap=16)
        assert len(chunks) >= 2

    def test_chunks_are_non_empty(self):
        from scripts.ingest import chunk_text
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text)
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_empty_text_returns_empty_list(self):
        from scripts.ingest import chunk_text
        chunks = chunk_text("", chunk_size=512, overlap=64)
        assert chunks == [] or all(not c.strip() for c in chunks)


# ── EvalQAPair schema validation ───────────────────────────────────────────────

class TestEvalQAPairSchema:

    def test_valid_pair(self):
        pair = EvalQAPair(
            id="qa-001",
            question="What did we learn?",
            ground_truth_doc_ids=["PM-001"],
            reference_answer="We learned XYZ.",
            difficulty="easy",
            category="post_mortem",
        )
        assert pair.id == "qa-001"
        assert pair.difficulty == "easy"

    def test_invalid_difficulty_raises(self):
        with pytest.raises(Exception):
            EvalQAPair(
                id="qa-bad",
                question="Q?",
                ground_truth_doc_ids=[],
                reference_answer="A.",
                difficulty="impossible",  # invalid Literal
                category="post_mortem",
            )
