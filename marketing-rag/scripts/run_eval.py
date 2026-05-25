"""
run_eval.py
Batch evaluation runner for the Marketing Insights RAG system.

Runs two separate eval loops:
  1. Retrieval eval (recall@k, MRR) — no LLM, cheap and fast
  2. Answer quality eval (LLM-as-judge) — uses Gemini Flash as judge

Results saved to reports/eval_summary.json

Run: python scripts/run_eval.py [--retrieval-only] [--n N]
"""

import asyncio
import json
import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.retriever import Retriever
from src.rag_engine import generate_answer
from src.eval_harness import (
    evaluate_retrieval,
    evaluate_answer_quality,
    summarize_retrieval_results,
    summarize_answer_results,
    CostTracker,
)
from src.models import EvalQAPair

EVAL_DATASET_PATH = Path(__file__).parent.parent / "eval_dataset" / "qa_pairs.json"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def load_eval_pairs(n: int | None = None) -> list[EvalQAPair]:
    with open(EVAL_DATASET_PATH) as f:
        raw = json.load(f)
    pairs = [EvalQAPair(**p) for p in raw]
    if n:
        pairs = pairs[:n]
    return pairs


async def run_answer_eval(
    qa_pairs: list[EvalQAPair],
    retriever: Retriever,
    cost_tracker: CostTracker,
) -> list[dict]:
    answer_results = []
    for i, qa in enumerate(qa_pairs):
        print(f"  [{i+1}/{len(qa_pairs)}] {qa.id}: {qa.question[:60]}...")
        try:
            chunks = retriever.retrieve(qa.question, top_n_rerank=5)
            answer, p_tok, c_tok = await generate_answer(qa.question, chunks)
            cost_tracker.add(p_tok, c_tok)
            result = await evaluate_answer_quality(qa, answer, chunks)
            answer_results.append(result)
            print(f"    ✓ faithfulness={result.faithfulness_score}, completeness={result.completeness_score}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
        await asyncio.sleep(0.5)  # rate limit buffer
    return answer_results


async def main(retrieval_only: bool = False, n: int | None = None):
    print("=" * 60)
    print("Marketing Insights RAG — Evaluation Suite")
    print("=" * 60)

    # Load eval pairs
    qa_pairs = load_eval_pairs(n)
    print(f"\n📋 Loaded {len(qa_pairs)} eval pairs\n")

    # Initialize retriever
    print("Initializing retriever...")
    retriever = Retriever()
    print("✅ Retriever ready\n")

    cost_tracker = CostTracker()
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_eval_pairs": len(qa_pairs),
    }

    # ── Phase 1: Retrieval Eval ────────────────────────────────────────────────
    print("─" * 40)
    print("Phase 1: Retrieval Eval (recall@k, MRR)")
    print("─" * 40)
    t0 = time.time()
    retrieval_results = []
    for i, qa in enumerate(qa_pairs):
        retrieved_ids = retriever.retrieve_for_eval(qa.question, top_k=10)
        result = evaluate_retrieval(qa, retrieved_ids)
        retrieval_results.append(result)
        r5 = result.recall_at_5
        mrr = result.reciprocal_rank
        print(f"  [{i+1}/{len(qa_pairs)}] {qa.id} | recall@5={r5:.2f} mrr={mrr:.2f}")

    retrieval_summary = summarize_retrieval_results(retrieval_results)
    report["retrieval_eval"] = {
        "summary": retrieval_summary,
        "per_query": [r.model_dump() for r in retrieval_results],
        "duration_s": round(time.time() - t0, 1),
    }

    print(f"\n📊 Retrieval Summary:")
    for k, v in retrieval_summary.items():
        print(f"   {k}: {v}")

    # ── Phase 2: Answer Quality Eval ──────────────────────────────────────────
    if not retrieval_only:
        print("\n" + "─" * 40)
        print("Phase 2: Answer Quality Eval (LLM-as-judge)")
        print("─" * 40)
        t1 = time.time()
        answer_results = await run_answer_eval(qa_pairs, retriever, cost_tracker)
        answer_summary = summarize_answer_results(answer_results)
        report["answer_quality_eval"] = {
            "summary": answer_summary,
            "per_query": [r.model_dump() for r in answer_results],
            "duration_s": round(time.time() - t1, 1),
        }
        report["cost"] = cost_tracker.summary()

        print(f"\n📊 Answer Quality Summary:")
        for k, v in answer_summary.items():
            print(f"   {k}: {v}")

        print(f"\n💰 Cost: {cost_tracker.summary()['estimated_cost_usd']}")

    # ── Save report ────────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "eval_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Report saved to {report_path}")
    print(f"   Total wall time: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip answer quality eval")
    parser.add_argument("--n", type=int, default=None, help="Number of eval pairs to run (default: all 30)")
    args = parser.parse_args()

    asyncio.run(main(retrieval_only=args.retrieval_only, n=args.n))
