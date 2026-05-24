from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import json
import time
import os
import logging

from .models import SKUState, CopilotResponse
from .llm_engine import get_pricing_recommendation
from .eval_harness import EvalHarness

# Configure standard logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Pricing Copilot API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze", response_model=Dict[str, Any])
async def analyze_dataset(file: UploadFile = File(...)):
    """
    Analyzes an uploaded JSON dataset of SKUs.
    Returns a list of recommendations and an evaluation summary.
    """
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON datasets are supported in this version.")
        
    content = await file.read()
    try:
        skus_data = json.loads(content.decode('utf-8'))
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
        
    # Limit to first 20 SKUs for the interactive demo to keep latency low
    skus_data = skus_data[:20]
    
    harness = EvalHarness()
    
    # Load baselines for evaluation scoring
    baselines = {}
    baseline_path = os.getenv("BASELINE_DATA_PATH", "data/baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baselines = json.load(f)
    else:
        logger.warning(f"Baseline file not found at {baseline_path}. Accuracy metrics may be skewed.")
        
    recommendations = []
    
    for sku_dict in skus_data:
        try:
            sku = SKUState(**sku_dict)
            baseline_move = baselines.get(sku.sku_id, "HOLD")
            
            start_time = time.time()
            rec, p_tokens, c_tokens = await get_pricing_recommendation(sku)
            latency_ms = int((time.time() - start_time) * 1000)
            
            harness.evaluate_prediction(
                input_state=sku,
                recommendation=rec,
                baseline_move=baseline_move,
                latency_ms=latency_ms,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens
            )
            # Enrich the recommendation payload with input context for the UI
            rec_dict = rec.model_dump()
            rec_dict["current_price"] = sku.current_price
            rec_dict["sku_name"] = sku.name
            rec_dict["category"] = sku.category
            rec_dict["inventory_level"] = sku.inventory_level
            rec_dict["sales_velocity_weekly"] = sku.sales_velocity_weekly
            recommendations.append(rec_dict)
        except Exception as e:
            logger.error(f"Failed processing SKU {sku_dict.get('sku_id')}: {e}")
            
    summary = harness.get_summary()
    logger.info(f"Successfully processed {len(recommendations)} SKUs with {summary.get('accuracy')} accuracy.")
    
    return {
        "recommendations": recommendations,
        "evaluation_summary": summary
    }

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
