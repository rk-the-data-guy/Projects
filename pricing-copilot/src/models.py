from pydantic import BaseModel, Field
from typing import List, Literal

class CompetitorData(BaseModel):
    competitor_name: str
    price: float
    in_stock: bool

class SKUState(BaseModel):
    sku_id: str
    name: str
    category: str
    cogs: float = Field(description="Cost of Goods Sold")
    current_price: float
    inventory_level: int
    sales_velocity_weekly: float = Field(description="Average units sold per week over the last 4 weeks")
    elasticity_estimate: float = Field(description="Estimated price elasticity of demand")
    competitor_pricing: List[CompetitorData]

class PricingRecommendation(BaseModel):
    sku_id: str
    recommended_move: Literal["INCREASE", "DECREASE", "HOLD"]
    new_price: float
    rationale: str = Field(description="Detailed explanation for the recommended price move")
    confidence_score: int = Field(ge=0, le=100, description="Confidence in this recommendation from 0 to 100")
    supporting_evidence: List[str] = Field(description="Specific data points that drove this decision")

class CopilotResponse(BaseModel):
    recommendations: List[PricingRecommendation]
