import pytest
from src.models import SKUState, PricingRecommendation
from src.eval_harness import calculate_hallucination_score, EvalHarness

def get_dummy_sku():
    return SKUState(
        sku_id="SKU-TEST",
        name="Test Item",
        category="Test",
        cogs=10.0,
        current_price=20.0,
        inventory_level=100,
        sales_velocity_weekly=5.0,
        elasticity_estimate=-1.5,
        competitor_pricing=[]
    )

def test_no_hallucinations():
    sku = get_dummy_sku()
    rec = PricingRecommendation(
        sku_id=sku.sku_id,
        recommended_move="HOLD",
        new_price=20.0,
        rationale="We have 100 units in inventory.",
        confidence_score=90,
        supporting_evidence=["Price is 20.0"]
    )
    score = calculate_hallucination_score(sku, rec)
    assert score == 0.0

def test_some_hallucinations():
    sku = get_dummy_sku()
    rec = PricingRecommendation(
        sku_id=sku.sku_id,
        recommended_move="INCREASE",
        new_price=25.0,
        rationale="We have 100 units. Competitor X is selling at 999.99.",
        confidence_score=80,
        supporting_evidence=[]
    )
    score = calculate_hallucination_score(sku, rec)
    # The valid numbers are: 10.0, 20.0, 100, 5.0, -1.5, and 25.0
    # Cited numbers in text: 100, 999.99
    # Hallucinated: 999.99
    # Score: 1 / 2 = 0.5
    assert score == 0.5

def test_eval_harness_accuracy():
    harness = EvalHarness()
    sku = get_dummy_sku()
    rec = PricingRecommendation(
        sku_id=sku.sku_id,
        recommended_move="INCREASE",
        new_price=25.0,
        rationale="Increasing to capture margin.",
        confidence_score=90,
        supporting_evidence=[]
    )
    
    # Test Match
    res_match = harness.evaluate_prediction(sku, rec, "INCREASE", 100, 50, 50)
    assert res_match.accuracy_score == 1.0
    
    # Test Mismatch
    rec.recommended_move = "DECREASE"
    res_mismatch = harness.evaluate_prediction(sku, rec, "INCREASE", 100, 50, 50)
    assert res_mismatch.accuracy_score == 0.0
