import os
import logging
from typing import Tuple
from google import genai
from google.genai import types
from dotenv import load_dotenv
from .models import SKUState, PricingRecommendation

# Load .env so GEMINI_API_KEY is available without manual export
load_dotenv()

logger = logging.getLogger(__name__)

# Note: Make sure to set GEMINI_API_KEY in the environment or .env file
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file (see .env.example).")

client = genai.Client(api_key=API_KEY)

PROMPT_V1 = """
You are an expert retail pricing analyst. Your goal is to review the current state of a SKU, 
including its inventory, sales velocity, elasticity, and competitor pricing, and recommend a pricing move.

Guidelines:
1. If we have high inventory (>300) and low sales (<10), we probably need to DECREASE price to move volume.
2. If we have low inventory (<20) and high sales (>30), we probably need to INCREASE price to capture margin.
3. Pay close attention to competitor pricing. If we are significantly more expensive than competitors, 
   we might be losing volume. If we are significantly cheaper, we are leaving margin on the table.
4. When providing rationale, ONLY cite numbers that are explicitly provided in the SKU data. Do not hallucinate numbers.
5. Provide a confidence score based on how strong the signals are.

Make a recommendation (INCREASE, DECREASE, or HOLD), provide the new proposed price, explain your rationale, 
and list the specific data points as supporting evidence.
"""

async def get_pricing_recommendation(sku: SKUState, prompt_version: str = "v1") -> Tuple[PricingRecommendation, int, int]:
    """
    Calls the Gemini API to get a pricing recommendation.
    Returns: (PricingRecommendation, prompt_tokens, completion_tokens)
    """
    system_prompt = PROMPT_V1
    
    sku_json = sku.model_dump_json(indent=2)
    user_message = f"Please analyze this SKU and provide a recommendation:\n{sku_json}"
    
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_text(system_prompt + "\n\n" + user_message)
                ])
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PricingRecommendation,
                temperature=0.1
            )
        )
        
        # Parse the structured JSON output into our Pydantic model
        recommendation = PricingRecommendation.model_validate_json(response.text)
        
        prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        completion_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        
        return recommendation, prompt_tokens, completion_tokens
        
    except Exception as e:
        logger.error(f"Error calling LLM for SKU {sku.sku_id}: {e}")
        raise e
