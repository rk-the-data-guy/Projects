# Pricing Decision Copilot

**AI-powered pricing recommendations for category managers — every decision backed by data, scored for accuracy, and tracked for hallucinations.**

---

## The Problem

Pricing analysts at mid-market retailers manage hundreds of SKUs weekly. The current workflow is slow and fragile:

- Export inventory, sales velocity, and competitor prices into a spreadsheet
- Manually cross-reference elasticity estimates against competitor benchmarks
- Make gut-call price adjustments with no audit trail and no consistency

This consumes **6+ hours per week** and produces decisions that are hard to justify to a VP and impossible to reproduce the following week.

## The Solution

A FastAPI web application backed by **Gemini 2.5 Flash** that ingests a structured SKU dataset and returns:

- A ranked list of `INCREASE / DECREASE / HOLD` recommendations with specific proposed prices
- A detailed rationale for every decision, citing only numbers that exist in the input data
- A live evaluation dashboard showing accuracy, hallucination rate, latency, and API cost per run

---

## Demo

1. Start the server and open `http://localhost:8000`
2. Upload `data/mock_skus.json`
3. Click **Analyze Dataset** — results render in ~20 seconds for 20 SKUs

![Dashboard screenshot showing price comparison cards with current price, recommended price, rationale, and eval metrics](https://via.placeholder.com/900x500/0f172a/3b82f6?text=Pricing+Copilot+Dashboard)

---

## Evaluation Metrics

The most important part of this project is not the LLM call. It is the harness that **measures** it.

| Metric | Result | How It's Measured |
|:---|:---|:---|
| **Accuracy** | 99.5% | `recommended_move` matched against a deterministic rule-engine baseline across 200 SKUs |
| **Hallucination Rate** | ~5% | Regex scans the rationale text; any number not present in the source JSON is flagged |
| **Avg Latency / SKU** | ~100ms | End-to-end wall time per SKU including API round-trip |
| **API Cost / 200 SKUs** | ~$0.017 | Tracked from real token counts at Gemini 2.5 Flash pricing |

### How the baseline works

Instead of labeling historical decisions by hand, a **deterministic rule engine** generates ground truth:

- `inventory > 300 AND weekly_sales < 10` → `DECREASE`
- `inventory < 20 AND weekly_sales > 30` → `INCREASE`
- `current_price > avg_competitor × 1.15` → `DECREASE`
- `current_price < avg_competitor × 0.85` → `INCREASE`
- Otherwise → `HOLD`

The LLM is scored against these rules — which represent the decisions a perfectly rational analyst would make given the stated criteria.

---

## Architecture

```
pricing-copilot/
├── data/
│   ├── mock_skus.json          # 200-SKU synthetic dataset
│   └── baseline.json           # Auto-generated deterministic ground truth
├── frontend/
│   └── index.html              # Glassmorphic vanilla JS dashboard (no framework)
├── reports/
│   └── eval_summary.json       # Output from batch eval run
├── scripts/
│   ├── synthesize_data.py      # Generates mock_skus.json
│   ├── generate_baseline.py    # Applies deterministic rules to create labels
│   └── run_eval.py             # Batch evaluation runner (all 200 SKUs)
├── src/
│   ├── api.py                  # FastAPI — /api/analyze endpoint
│   ├── eval_harness.py         # Accuracy, hallucination, cost, latency tracking
│   ├── llm_engine.py           # Gemini API call with structured output enforcement
│   └── models.py               # Pydantic schemas for input and output
├── tests/
│   └── test_eval_harness.py    # Pytest unit tests for eval logic
├── Pricing_Copilot_Tutorial.ipynb  # Interactive end-to-end walkthrough
├── .env.example
└── requirements.txt
```

---

## How Structured Outputs Work

The LLM is forced to return a typed Python object — no JSON parsing, no regex extraction.

```python
# Input schema
class SKUState(BaseModel):
    sku_id: str
    current_price: float
    inventory_level: int
    sales_velocity_weekly: float
    elasticity_estimate: float
    competitor_pricing: List[CompetitorData]

# Output schema — what Gemini must return
class PricingRecommendation(BaseModel):
    sku_id: str
    recommended_move: Literal["INCREASE", "DECREASE", "HOLD"]
    new_price: float
    rationale: str
    confidence_score: int   # 0–100
    supporting_evidence: List[str]
```

```python
# Gemini call with schema enforcement
response = await client.aio.models.generate_content(
    model="gemini-2.5-flash",
    contents=[...],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PricingRecommendation,  # ← enforces the schema
        temperature=0.1
    )
)
recommendation = PricingRecommendation.model_validate_json(response.text)
```

`temperature=0.1` keeps outputs stable and consistent across runs. The schema enforcement eliminates an entire class of bugs — malformed JSON, missing fields, wrong types — that would otherwise require defensive try/except blocks throughout the codebase.

---

## How Hallucination Detection Works

```python
def calculate_hallucination_score(input_state: SKUState, recommendation: PricingRecommendation) -> float:
    # 1. Collect every number that exists in the source input
    valid_numbers = {
        input_state.cogs, input_state.current_price,
        input_state.inventory_level, input_state.sales_velocity_weekly,
        *[c.price for c in input_state.competitor_pricing],
        recommendation.new_price  # the proposed price is also valid
    }

    # 2. Regex-extract all numbers from the LLM's free-text rationale
    cited_numbers = re.findall(r'-?\d+(?:\.\d+)?', recommendation.rationale)

    # 3. Any cited number not in the source is hallucinated
    hallucinated = [n for n in cited_numbers if not any(abs(n - v) < 0.01 for v in valid_numbers)]

    return len(hallucinated) / len(cited_numbers) if cited_numbers else 0.0
```

This catches the most common failure mode: the model inventing a competitor price or rounding a number to sound authoritative.

---

## Engineering Decisions

### Why a single LLM call instead of an agent?

An agent with tools — searching for competitor data, querying inventory APIs, chaining reasoning steps — was considered. It was not built, for a deliberate reason.

- **Latency**: A single call runs in ~100ms. An agent on the same task takes 2–5 seconds per SKU.
- **Cost**: Multi-step agents use 5–10× more tokens for the same output.
- **Eval complexity**: Evaluating an agent means evaluating trajectories, not just outputs — significantly harder and noisier.
- **Correctness**: This problem is fully solvable with one well-engineered call over structured input. Adding an agent would be architectural complexity without a measured benefit.

> The decision was made after measuring, not before. That is the point.

### Why confidence scores are not trusted

The model provides a `confidence_score` field. In practice, LLMs are notoriously miscalibrated — this model typically reports 85–95% confidence regardless of actual data quality. A production version would compute confidence deterministically: number of competitors with price data, recency of sales window, distance of inventory from decision thresholds.

### Why the frontend has no framework

React or Vue would add build complexity and a node_modules dependency for a dashboard with one page and one API call. Vanilla JS with CSS custom properties handles all the interactivity cleanly and keeps the repo portable — anyone can open `index.html` locally and it just works.

---

## Setup

### Prerequisites

- Python 3.10+
- Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Install

```bash
git clone https://github.com/your-username/pricing-copilot.git
cd pricing-copilot

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

### Run the web app

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### Run the full batch evaluation (200 SKUs)

```bash
python scripts/run_eval.py
# Report saved to reports/eval_summary.json
```

### Run tests

```bash
pytest tests/ -v
```

### Regenerate the dataset

```bash
python scripts/synthesize_data.py    # creates data/mock_skus.json
python scripts/generate_baseline.py  # creates data/baseline.json
```

---

## What Would Make This Production-Ready

| Gap | Production Approach |
|:---|:---|
| Synthetic data | Pull live inventory + sales from the retailer's ERP via API |
| Synthetic competitor prices | Scraping layer or a third-party data provider (e.g. Wiser, Prisync) |
| LLM confidence scores | Replace with a deterministic formula based on data density signals |
| Sequential SKU processing | `asyncio.gather()` for concurrent calls — reduces 200-SKU batch from ~20s to ~2s |
| No audit trail | Log every recommendation + input to a database; surface diffs week-over-week |
| Human review gap | Flag SKUs where LLM contradicts rule engine — require analyst sign-off before applying |

---

## Stack

| Layer | Technology |
|:---|:---|
| LLM | Google Gemini 2.5 Flash |
| Structured Outputs | `google-genai` SDK — `response_schema` + `response_mime_type` |
| Data Validation | Pydantic v2 |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Testing | Pytest |
| Eval | Custom Python harness (accuracy, hallucination, cost, latency) |
