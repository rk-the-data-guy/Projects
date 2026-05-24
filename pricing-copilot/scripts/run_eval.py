import json
import asyncio
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SKUState
from src.llm_engine import get_pricing_recommendation
from src.eval_harness import EvalHarness

async def run_evaluation():
    print("Loading data...")
    with open("data/mock_skus.json", "r") as f:
        skus_data = json.load(f)
        
    with open("data/baseline.json", "r") as f:
        baselines = json.load(f)
        
    harness = EvalHarness()
    
    print(f"Running evaluation on {len(skus_data)} SKUs...")
    
    for i, sku_dict in enumerate(skus_data):
        sku = SKUState(**sku_dict)
        baseline_move = baselines[sku.sku_id]
        
        start_time = time.time()
        rec, prompt_tokens, comp_tokens = await get_pricing_recommendation(sku)
        latency_ms = int((time.time() - start_time) * 1000)
        
        harness.evaluate_prediction(
            input_state=sku,
            recommendation=rec,
            baseline_move=baseline_move,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens
        )
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(skus_data)} SKUs...")
            
    summary = harness.get_summary()
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n--- Evaluation Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("--------------------------")
    print("Full report saved to reports/eval_summary.json")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
