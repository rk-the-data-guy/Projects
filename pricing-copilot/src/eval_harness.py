import re
from typing import Dict, Any, List
from .models import SKUState, PricingRecommendation

# Example costs for Gemini 2.5 Flash
COST_PER_1K_PROMPT_TOKENS = 0.000075
COST_PER_1K_SAMPLED_TOKENS = 0.00030

class EvaluationResult:
    """
    Stores the evaluation metrics for a single SKU prediction, 
    including accuracy, hallucination scores, latency, and cost.
    """
    def __init__(self, sku_id: str):
        self.sku_id = sku_id
        self.latency_ms = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        
        self.hallucination_score = 0.0 # 0.0 means no hallucinated numbers, 1.0 means all numbers are hallucinated
        self.accuracy_score = 0.0 # 1.0 if recommended move matches baseline, 0.0 otherwise
        
        self.errors = []
        self.baseline_matched = False

def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from a string as a list of floats."""
    raw_numbers = re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
    return [float(n) for n in raw_numbers]

def calculate_hallucination_score(input_state: SKUState, recommendation: PricingRecommendation) -> float:
    """
    Check if numbers mentioned in the rationale and evidence actually exist in the input state.
    Returns a score between 0.0 (no hallucinations) and 1.0 (all numbers hallucinated).
    """
    valid_numbers = set()
    valid_numbers.add(float(input_state.cogs))
    valid_numbers.add(float(input_state.current_price))
    valid_numbers.add(float(input_state.inventory_level))
    valid_numbers.add(float(input_state.sales_velocity_weekly))
    valid_numbers.add(float(input_state.elasticity_estimate))
    
    for comp in input_state.competitor_pricing:
        valid_numbers.add(float(comp.price))
        
    valid_numbers.add(float(recommendation.new_price))

    output_text = recommendation.rationale + " " + " ".join(recommendation.supporting_evidence)
    cited_numbers = extract_numbers(output_text)
    
    if not cited_numbers:
        return 0.0
        
    hallucinated_count = 0
    for num in cited_numbers:
        matched = False
        for valid_num in valid_numbers:
            if abs(num - valid_num) < 0.01:
                matched = True
                break
        if not matched:
            if num not in [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]:
                hallucinated_count += 1
                
    return hallucinated_count / len(cited_numbers)


class EvalHarness:
    """
    The core evaluation framework that runs regression tests against LLM predictions.
    It tracks accuracy against a baseline, hallucination rates, latency, and API costs.
    """
    def __init__(self):
        self.results = []
        
    def evaluate_prediction(
        self, 
        input_state: SKUState, 
        recommendation: PricingRecommendation, 
        baseline_move: str, 
        latency_ms: int,
        prompt_tokens: int,
        completion_tokens: int
    ) -> EvaluationResult:
        
        result = EvaluationResult(input_state.sku_id)
        result.latency_ms = latency_ms
        result.prompt_tokens = prompt_tokens
        result.completion_tokens = completion_tokens
        
        prompt_cost = (prompt_tokens / 1000) * COST_PER_1K_PROMPT_TOKENS
        completion_cost = (completion_tokens / 1000) * COST_PER_1K_SAMPLED_TOKENS
        result.total_cost = prompt_cost + completion_cost
        
        result.hallucination_score = calculate_hallucination_score(input_state, recommendation)
        
        if recommendation.recommended_move == baseline_move:
            result.accuracy_score = 1.0
            result.baseline_matched = True
        else:
            result.accuracy_score = 0.0
            result.baseline_matched = False
            
        self.results.append(result)
        return result
        
    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No results to summarize"}
            
        total_evals = len(self.results)
        avg_latency = sum(r.latency_ms for r in self.results) / total_evals
        total_cost = sum(r.total_cost for r in self.results)
        avg_hallucination = sum(r.hallucination_score for r in self.results) / total_evals
        accuracy = sum(r.accuracy_score for r in self.results) / total_evals
        
        return {
            "total_evals": total_evals,
            "accuracy": f"{accuracy * 100:.1f}%",
            "avg_hallucination_score": f"{avg_hallucination * 100:.1f}%",
            "avg_latency_ms": round(avg_latency, 2),
            "total_cost_usd": f"${total_cost:.4f}",
        }
