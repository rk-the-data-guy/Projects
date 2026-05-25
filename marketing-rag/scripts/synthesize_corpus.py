"""
synthesize_corpus.py
Generate 520 realistic brand marketing documents across 5 categories.
Documents are structured so the eval Q&A pairs can be reliably answered from them.
Low-index documents (indexes 1-7) are deterministic to match qa_pairs.json ground truths.
All other documents are generated randomly as background noise.

Output: data/corpus/<doc_id>.json  (each file is a JSON doc with metadata + body)

Run: python scripts/synthesize_corpus.py
"""

import json
import os
import random
from datetime import datetime, timedelta

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")

# ── Seed data ──────────────────────────────────────────────────────────────────

BRANDS = ["Luminary", "NovaNest", "PeakFlow", "Horizon", "Ember", "Clarity", "Drift"]
CAMPAIGNS = [
    "Back-to-School 2023", "Holiday Blitz 2023", "Summer Recharge 2023",
    "Valentine's Launch 2024", "Spring Refresh 2024", "Earth Day Push 2024",
    "Back-to-School 2022", "Black Friday 2023", "New Year New You 2024",
    "Mother's Day 2024", "Father's Day 2024", "Cyber Monday 2023",
    "Pride Month 2024", "Labor Day 2023", "Memorial Day 2023",
]
CHANNELS = ["Instagram", "TikTok", "Email", "Google Ads", "Pinterest", "YouTube", "Influencer", "OOH"]
AUDIENCES = [
    "Gen Z (18–24)", "Millennials (25–34)", "Parents (30–45)", "Professionals (28–45)",
    "College Students", "Health-conscious consumers", "Premium shoppers",
]
COMPETITORS = ["BrandX", "Nimbus Co.", "Elevate", "RootWorks", "PulseMedia"]
PERSONAS = [
    {"name": "The Achiever", "age": "28–35", "income": "85k–120k", "values": "performance, convenience"},
    {"name": "The Conscious Consumer", "age": "22–32", "income": "45k–70k", "values": "sustainability, authenticity"},
    {"name": "The Trendsetter", "age": "18–26", "income": "30k–55k", "values": "novelty, social proof"},
    {"name": "The Loyalist", "age": "35–50", "income": "90k–150k", "values": "reliability, quality"},
    {"name": "The Deal Hunter", "age": "25–45", "income": "40k–65k", "values": "value, practicality"},
]


def random_date(start_year=2022, end_year=2024) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return (start + timedelta(days=random.randint(0, (end - start).days))).strftime("%Y-%m-%d")


def make_doc_id(category: str, index: int) -> str:
    prefix = {
        "campaign_brief": "CB",
        "post_mortem": "PM",
        "customer_research": "CR",
        "win_loss": "WL",
        "slack_thread": "ST",
    }[category]
    return f"{prefix}-{index:04d}"


# ── Deterministic Documents for Low Indexes ───────────────────────────────────

def get_low_index_campaign_brief(index: int) -> dict:
    if index == 1:
        campaign = "Back-to-School 2023"
        brand = "Horizon"
        date = "2023-06-15"
        channels = ["TikTok", "Instagram", "Email"]
        audience = "Parents (30–45)"
        budget = 150000
        body = f"""# Campaign Brief: Back-to-School 2023 — Horizon

**Date:** 2023-06-15
**Author:** Marketing Strategy Team
**Campaign:** Back-to-School 2023
**Brand:** Horizon

## Objective
Drive acquisition at target CPA of $45. Target ROAS for paid campaigns is set at 4.5x on paid channels.

## Target Audience
Parents (30–45).

## Channels & Budget Allocation
Total budget: $150,000
- **TikTok**: 40% of budget
- **Instagram**: 35% of budget
- **Email**: 25% of budget

## Creative Direction
Lead with UGC-style short-form video on TikTok. Secondary assets are static carousels emphasizing the product benefit.

## Messaging Framework
- Primary message: "Built different."

## Risk Flags
- Competitor BrandX is running a similar promotion; monitor weekly.
- Creative fatigue risk on TikTok after week 2 — have refresh assets ready.
- Offer mechanics leak risk like coupon leaks.

## Success Metrics
- Target ROAS: 4.5x
- Email Open Rate: Target of 32%
"""
    elif index == 2:
        campaign = "Back-to-School 2022"
        brand = "PeakFlow"
        date = "2022-06-10"
        channels = ["TikTok", "Instagram", "Email"]
        audience = "Gen Z (18–24)"
        budget = 120000
        body = f"""# Campaign Brief: Back-to-School 2022 — PeakFlow

**Date:** 2022-06-10
**Author:** Marketing Strategy Team
**Campaign:** Back-to-School 2022
**Brand:** PeakFlow

## Objective
Establish strong presence. Target CPA is $50. Target ROAS is 4.0x.

## Target Audience
Gen Z (18–24).

## Channels & Budget Allocation
Recommended Instagram vs TikTok budget allocation approach: TikTok has consistently outperformed Instagram. Therefore, we increase TikTok allocation while treating Instagram as the secondary channel for awareness, allocating a smaller share of budget to Instagram.
Total budget: $120,000
- **TikTok**: 55% of budget
- **Instagram**: 25% of budget
- **Email**: 20% of budget

## Creative Direction
Micro-creator influencer partnerships ROI is expected to exceed brand-paid ads, so focus budget here.

## Success Metrics
- Email Open Rate: Target is 28%
"""
    elif index == 3:
        campaign = "Summer Recharge 2023"
        brand = "Ember"
        date = "2023-05-12"
        channels = ["TikTok", "Instagram"]
        audience = "Premium shoppers"
        budget = 180000
        body = f"""# Campaign Brief: Summer Recharge 2023 — Ember

**Date:** 2023-05-12
**Campaign:** Summer Recharge 2023
**Brand:** Ember

## Objective
Target ROAS for our paid campaigns is set between 2.5x and 6.0x depending on channel. For this premium campaign, the target ROAS is 3.5x.

## Risk Flags
- Creative fatigue risk on Instagram. The template includes a dedicated Risk Flags section covering competitor activity (share of voice), creative fatigue (refresh asset requirements), and coupon leaks.
"""
    elif index == 4:
        campaign = "Valentine's Launch 2024"
        brand = "Clarity"
        date = "2024-01-05"
        channels = ["Instagram", "Email"]
        audience = "Millennials (25–34)"
        budget = 200000
        body = f"""# Campaign Brief: Valentine's Launch 2024 — Clarity

**Date:** 2024-01-05
**Campaign:** Valentine's Launch 2024
**Brand:** Clarity

## Objective
Valentine's Launch strategy: Drive new customer acquisition with UGC-style creative on social channels (Instagram) and strong email segmentation. Paid social allocates heavily to Instagram with warm, authentic conversion-focused CTAs.
"""
    elif index == 5:
        campaign = "Spring Refresh 2024"
        brand = "Luminary"
        date = "2024-03-01"
        channels = ["Email", "Instagram"]
        audience = "Parents (30–45)"
        budget = 90000
        body = f"""# Campaign Brief: Spring Refresh 2024 — Luminary

**Date:** 2024-03-01
**Campaign:** Spring Refresh 2024
**Brand:** Luminary

## Objective
Target ROAS is 5.0x.

## Risk Flags
- dedicated Risk Flags section: competitor activity monitoring, creative fatigue (refresh asset requirements), and coupon leak mitigation.

## Success Metrics
- Target Email Open Rate: 35%
"""
    elif index == 6:
        campaign = "Valentine's Launch 2024"
        brand = "Horizon"
        date = "2024-01-10"
        channels = ["Instagram", "Email"]
        audience = "Millennials (25–34)"
        budget = 160000
        body = f"""# Campaign Brief: Valentine's Launch 2024 — Horizon

**Date:** 2024-01-10
**Campaign:** Valentine's Launch 2024
**Brand:** Horizon

## Objective
Valentine's Launch strategy focuses on new customer acquisition using UGC-style creative on social channels and strong email segmentation. Warm, authentic messaging with conversion-focused CTAs.
"""
    else:
        return {}

    return {
        "doc_id": make_doc_id("campaign_brief", index),
        "title": f"Campaign Brief: {campaign} — {brand}",
        "doc_type": "campaign_brief",
        "campaign": campaign,
        "brand": brand,
        "date": date,
        "channels": channels,
        "audience": audience,
        "budget_usd": budget,
        "body": body,
    }


def get_low_index_post_mortem(index: int) -> dict:
    if index == 1:
        campaign = "Back-to-School 2023"
        brand = "Horizon"
        date = "2023-09-10"
        channels = ["TikTok", "Instagram", "Email", "Google Ads"]
        body = f"""# Post-Mortem: Back-to-School 2023 — Horizon

**Date:** 2023-09-10
**Campaign:** Back-to-School 2023
**Brand:** Horizon

## What Worked
- **Email Channel**: Email segmentation by purchase history consistently delivered the lowest CPA across campaigns, driving a major share of revenue at lower cost.
- **Creative Format**: UGC-style short-form video (under 15 seconds) was the best performing creative format in back-to-school and summer campaigns, outperforming produced video by up to 3.2x on CTR. Static carousels with bold headlines were second.
- **TikTok**: TikTok was a surprise hit and way outperformed Instagram. Recommendation is to increase TikTok budget allocation.

## What Didn't Work
- **Creative Fatigue**: Creative fatigue sets in quickly—typically by day 4-5—causing CTR to drop 40-70% week-over-week. We need to build a creative refresh buffer (e.g. 48-hour) and have assets ready before launch.
- **Discount Code Leak**: Coupon code leaked to deal forums, attracting 30% deal-seeking customers who purchased once and returned at 3x the average return rate, inflating spend. Recommend gated offers and single-use codes.
- **Broad Match Keyword Failure**: Broad match keywords on Google Ads drove unqualified traffic with poor conversion. Use exact match from the start.
- **Late Email Launch**: Delays in email launch (due to 3-7 day legal review cycles) cost us the opening weekend window, leaving 25% of revenue on the table.
"""
    elif index == 2:
        campaign = "Holiday Blitz 2023"
        brand = "PeakFlow"
        date = "2024-01-05"
        channels = ["OOH", "Instagram", "Pinterest", "Google Ads", "Influencer"]
        body = f"""# Post-Mortem: Holiday Blitz 2023 — PeakFlow

**Date:** 2024-01-05
**Campaign:** Holiday Blitz 2023
**Brand:** PeakFlow

## Key Learnings & Top 3 Learnings
1. **Creative Refresh Buffer**: Creative refresh timeline was reactive. We need a minimum of 2 weeks of buffer with pre-built refresh assets ready to deploy.
2. **Audience Segmentation**: Broad audience groupings underperform. Treating a heterogeneous audience as one segment (over-broad audience groupings) failed. Segments must be treated distinctly (e.g. Gen Z vs Millennials).
3. **Competitor Monitoring**: Competitor monitoring must be weekly, not monthly. Competitors adjust pricing mid-campaign.

## Mid-Campaign Budget Reallocation lag
Mid-campaign budget reallocation requires a 3-day lead time for platform algorithm re-learning. Teams must not reallocate without checking with leadership first.

## Influencer Collab ROI
Micro-creator influencer partnerships delivered 35% lower CPA than paid social ads. ROI exceeded expectations; double down next time.

## Attribution Model Disagreement
Attribution model disagreements between channels inflated our ROAS estimate by approximately 15%. Need a unified Touch attribution model.
"""
    elif index == 3:
        campaign = "Summer Recharge 2023"
        brand = "Ember"
        date = "2023-08-28"
        channels = ["Email", "Google Ads", "Instagram"]
        body = f"""# Post-Mortem: Summer Recharge 2023 — Ember

**Date:** 2023-08-28
**Campaign:** Summer Recharge 2023
**Brand:** Ember

## Insights
- **Lowest CPA**: Email segmentation by purchase history delivered the lowest CPA across campaigns.
- **Coupon Code Leak**: Leaks to deal forums attracted 22% deal-seekers who returned at 3x the average return rate. Gated offers are needed.
- **Best Creative Format**: UGC-style short-form video outperformed produced video by 3.0x on CTR.
- **Late Email Launch**: Launch delay of 5 days (due to copy review) cost the opening weekend, leaving 18% of revenue on the table.
- **Broad Match Keyword Failure**: Broad match on Google Ads drove unqualified traffic. Start with exact match.
"""
    elif index == 4:
        campaign = "Holiday Blitz 2023"
        brand = "Horizon"
        date = "2024-01-08"
        channels = ["Influencer", "Instagram"]
        body = f"""# Post-Mortem: Holiday Blitz 2023 — Horizon

**Date:** 2024-01-08
**Campaign:** Holiday Blitz 2023

## Learnings
- **Influencer ROI**: Micro-creator influencer partnerships achieved 25% lower CPA than brand-paid social ads. ROI exceeded expectations.
- **Audience Segmentation**: Over-broad audience groupings resulted in messaging that resonated with no sub-group strongly.
- **Attribution Model Disagreement**: Model disagreements inflated reported ROAS by 15%.
"""
    elif index == 5:
        campaign = "New Year New You 2024"
        brand = "Clarity"
        date = "2024-02-15"
        channels = ["Google Ads", "Email"]
        body = f"""# Post-Mortem: New Year New You 2024 — Clarity

**Date:** 2024-02-15

## Key Findings
- **Lowest CPA**: Email segmentation by purchase history drove the lowest CPA.
- **Broad Match Keyword Failure**: Google Ads broad match drove unqualified traffic with poor conversion; recommend exact match.
"""
    elif index == 6:
        campaign = "Spring Refresh 2024"
        brand = "Luminary"
        date = "2024-05-10"
        channels = ["OOH", "Instagram"]
        body = f"""# Post-Mortem: Spring Refresh 2024 — Luminary

**Date:** 2024-05-10

## Insights
- **Attribution Disagreement**: Attribution model disagreements between channels inflated ROAS estimate by 15%.
"""
    elif index == 7:
        campaign = "Earth Day Push 2024"
        brand = "Ember"
        date = "2024-05-05"
        channels = ["Email", "TikTok"]
        body = f"""# Post-Mortem: Earth Day Push 2024 — Ember

**Date:** 2024-05-05

## Insights
- **Lowest CPA**: Email segmentation delivered lowest CPA.
"""
    else:
        return {}

    return {
        "doc_id": make_doc_id("post_mortem", index),
        "title": f"Post-Mortem: {campaign} — {brand}",
        "doc_type": "post_mortem",
        "campaign": campaign,
        "brand": brand,
        "date": date,
        "channels": channels,
        "body": body,
    }


def get_low_index_customer_research(index: int) -> dict:
    if index == 1:
        persona = "The Achiever"
        brand = "NovaNest"
        date = "2023-11-15"
        body = f"""# Customer Research Report: The Achiever Segment — NovaNest

**Date:** 2023-11-15
**Brand:** NovaNest
**Segment:** The Achiever

## Executive Summary
The Achiever segment represents 25% of our customer base but contributes 42% of revenue. NPS score range for this segment is 55 (ranges from 22 to 72).

## Lapse & Reactivation
- **Lapse Drivers**: Price sensitivity and email fatigue are primary lapse drivers.
- **Reactivation**: Reactivation email campaigns using personalized offers based on past purchase history achieved a 15% reactivation rate (falls in the 8-22% range) when triggered at the 90-day mark.
- **Quality Signaling Pricing Power**: Pricing architecture study shows 32% of respondents (in 28-42% range) would pay 15% more (in 10-25% range) if quality signaling was clearer.

## Brand Perception
Descriptors most commonly associated with our brands: "reliable", "innovative", "premium", "trustworthy", and "expensive" (often paired with "worth it").
"""
    elif index == 2:
        persona = "The Conscious Consumer"
        brand = "Horizon"
        date = "2023-10-20"
        body = f"""# Customer Research Report: The Conscious Consumer Segment — Horizon

**Date:** 2023-10-20
**Brand:** Horizon
**Segment:** The Conscious Consumer

## Messaging Resonance
- **Sustainability & Authenticity**: Messaging concepts emphasizing sustainability, authenticity, and real results outperform generic category messaging by 25-55%.
- **Luxury For Less**: Concepts like "luxury for less" or trend-chasing underperformed.

## Lapse & Reactivation
Lapse drivers: Price sensitivity, email fatigue, and competitor pricing. Reactivation email campaign achieved 18% reactivation rate when triggered with personalized offers at 90 days.

## Quality Signaling
38% of Conscious Consumers would pay 20% more if quality signaling was clearer.

## Brand Perception
Words most commonly associated with our brands: "reliable", "innovative", "premium", "trustworthy", and "expensive" (but "worth it").
"""
    elif index == 3:
        persona = "The Loyalist"
        brand = "Ember"
        date = "2023-12-05"
        body = f"""# Customer Research Report: The Loyalist Segment — Ember

**Date:** 2023-12-05
**Segment:** The Loyalist

## Insights
- **Size & Revenue**: Represents 28% of customer base and 45% of revenue.
- **NPS Range**: NPS score range is 65 (ranging from 22 to 72).
- **Reactivation**: Achieved 10% reactivation rate using personalized offers at 90 days.
- **Quality Signaling**: 30% of Loyalists would pay 15% more if quality signaling was clearer.
- **Brand Perception Descriptors**: "reliable", "innovative", "premium", "trustworthy", "expensive".
"""
    elif index == 4:
        persona = "The Conscious Consumer"
        brand = "Clarity"
        date = "2024-02-18"
        body = f"""# Customer Research Report: The Conscious Consumer Segment — Clarity

**Date:** 2024-02-18

## Insights
- **Messaging Resonance**: Sustainability, authenticity, and real results outperform luxury for less by 40%.
- **Lapse reasons**: Price sensitivity, product not meeting expectations, email fatigue, competitors.
- **Quality Signaling**: 40% would pay 10% more if quality signaling was clearer.
- **Brand Perception Descriptors**: "reliable", "innovative", "premium", "trustworthy", "expensive".
"""
    elif index == 5:
        persona = "The Trendsetter"
        brand = "Luminary"
        date = "2024-03-22"
        body = f"""# Customer Research Report: The Trendsetter Segment — Luminary

**Date:** 2024-03-22

## Insights
- **Lapse reasons**: found better alternative, price sensitivity, email fatigue, competitor promotions.
- **Brand Perception Descriptors**: "reliable", "innovative", "premium", "trustworthy", "expensive".
"""
    elif index == 6:
        persona = "The Conscious Consumer"
        brand = "NovaNest"
        date = "2024-04-10"
        body = f"""# Customer Research Report: The Conscious Consumer Segment — NovaNest

**Date:** 2024-04-10

## Insights
- **Messaging Resonance**: Sustainability, authenticity, and real results outperform luxury for less by 35%.
"""
    else:
        return {}

    return {
        "doc_id": make_doc_id("customer_research", index),
        "title": f"Customer Research: {persona} Segment — {brand}",
        "doc_type": "customer_research",
        "brand": brand,
        "persona": persona,
        "date": date,
        "body": body,
    }


def get_low_index_win_loss(index: int) -> dict:
    dates = {
        1: "2023-09-05",
        2: "2023-08-14",
        3: "2023-11-20",
        4: "2023-10-12",
        5: "2023-12-10",
        6: "2024-01-18",
        7: "2024-02-05"
    }
    date = dates.get(index, "2023-09-05")
    if index == 1:
        competitor = "BrandX"
        brand = "Luminary"
        won = False
        body = f"""# Win/Loss Note: Luminary vs. BrandX

**Date:** 2023-09-05
**Outcome:** LOSS ❌
**Brand:** Luminary
**Competitor:** BrandX

## Primary Reason We Lost competitive deals
We lost competitive deals to BrandX due to:
1. **Aggressive Discounting**: Competitors discounted aggressively, offering 15-35% off list price (BrandX offered 25% discount).
2. **Pre-existing Relationships**: Prospects had pre-existing relationships with competitors.
3. **Feature & Compliance Gaps**: Lacked specific compliance certifications or feature gaps that we couldn't close.

## Decision Criteria (Ranked by Prospect)
1. **Integration capabilities and ease of use**
2. **Support quality and onboarding experience**
3. **Pricing and contract flexibility**
4. **Customer references and case studies**
"""
    elif index == 2:
        competitor = "Nimbus Co."
        brand = "NovaNest"
        won = False
        body = f"""# Win/Loss Note: NovaNest vs. Nimbus Co.

**Date:** 2023-08-14
**Outcome:** LOSS ❌
**Competitor:** Nimbus Co.

## Hurdle: Pre-existing relationships
Prospect had a pre-existing relationship with Nimbus Co.
- **Recommended Approach**: Multi-threading relationship-first accounts starting 6+ months before the evaluation begins. Competing purely on features has low win rate.

## Decision Criteria (Ranked by Prospect)
1. **Integration capabilities and ease of use**
2. **Support quality and onboarding experience**
3. **Pricing and contract flexibility**
4. **Customer references and case studies**
"""
    elif index == 3:
        competitor = "BrandX"
        brand = "PeakFlow"
        won = False
        body = f"""# Win/Loss Note: PeakFlow vs. BrandX

**Date:** 2023-11-20
**Outcome:** LOSS ❌

## Compliance Gaps
We lost because we lacked a specific compliance certification that BrandX holds. Compliance gaps have caused us to lose competitive deals in regulated verticals (financial services and healthcare-adjacent accounts).

## Pricing Intel
Competitor BrandX is discounting aggressively to win logos—offering 20% off list price.
"""
    elif index == 4:
        competitor = "Elevate"
        brand = "Horizon"
        won = False
        body = f"""# Win/Loss Note: Horizon vs. Elevate

**Date:** 2023-10-12
**Outcome:** LOSS ❌

## Insights
Prospect had pre-existing relationship with Elevate. Recommended approach when a prospect has a pre-existing relationship is to multi-thread these accounts starting 6+ months before evaluation.
- **Decision Criteria**: (1) Integration capabilities & ease of use, (2) Support quality & onboarding, (3) Pricing & contract flexibility, (4) Customer references.
"""
    elif index == 5:
        competitor = "BrandX"
        brand = "Ember"
        won = False
        body = f"""# Win/Loss Note: Ember vs. BrandX

**Date:** 2023-12-10
**Outcome:** LOSS ❌

## Reasons Lost
Lost to BrandX due to aggressive discounting (30% off list price), existing competitor relationships, and specific compliance gaps. Compliance certifications are sales-critical in regulated financial services and healthcare-adjacent verticals.
"""
    elif index == 6:
        competitor = "RootWorks"
        brand = "Clarity"
        won = False
        body = f"""# Win/Loss Note: Clarity vs. RootWorks

**Date:** 2024-01-18
**Outcome:** LOSS ❌

## Insights
Pre-existing relationship with competitor. Recommended approach is multi-threading these accounts 6+ months before evaluation begins.
- **Decision Criteria**: (1) Integration capabilities & ease of use, (2) Support quality & onboarding, (3) Pricing & contract flexibility, (4) Customer references.
"""
    elif index == 7:
        competitor = "BrandX"
        brand = "Luminary"
        won = False
        body = f"""# Win/Loss Note: Luminary vs. BrandX

**Date:** 2024-02-05
**Outcome:** LOSS ❌

## Insights
We lacked a specific compliance certification that BrandX holds, blocking regulated vertical deals. Competitor discounted aggressively at 15-35%.
"""
    else:
        return {}

    return {
        "doc_id": make_doc_id("win_loss", index),
        "title": f"Win/Loss: {brand} vs. {competitor} (Loss)",
        "doc_type": "win_loss",
        "brand": brand,
        "competitor": competitor,
        "date": date,
        "won": won,
        "body": body,
    }


def get_low_index_slack_thread(index: int) -> dict:
    if index == 1:
        campaign = "Father's Day 2024"
        brand = "NovaNest"
        date = "2024-06-20"
        participants = ["Casey", "Alex", "Drew", "Morgan", "Riley"]
        body = f"""# Slack Thread Summary: Quick sync on Father's Day 2024 performance

**Date:** 2024-06-20
**Channel:** #marketing-novanest
**Participants:** Casey, Alex, Drew, Morgan, Riley
**Campaign Context:** Father's Day 2024

## Thread Summary
Casey: Mid-campaign budget reallocation requires a 3-day lead time for platform algorithm re-learning. Do not reallocate without checking with leadership first and building this lag into campaign plans.
Morgan: Yes, algorithm re-learning takes exactly 3 days.

## Recurring Action Items from campaign Slack retrospectives
- **Draft post-mortem** capturing top learnings.
- **Send creative brief** to design with 48-hour timeline.
- **Update audience segment definitions**.
- **Monitor competitor pricing** weekly.
- **Update campaign tracker** with final metrics.
"""
    elif index == 2:
        campaign = "Holiday Blitz 2023"
        brand = "PeakFlow"
        date = "2024-01-02"
        participants = ["Alex", "Jamie", "Morgan", "Casey", "Riley"]
        body = f"""# Slack Thread Summary: Lessons from Holiday Blitz 2023 — what should we do differently?

**Date:** 2024-01-02
**Channel:** #marketing-peakflow
**Participants:** Alex, Jamie, Morgan, Casey, Riley
**Campaign Context:** Holiday Blitz 2023

## Thread Summary
Alex: Lessons from Holiday Blitz:
1. **Creative Refresh Timeline**: We were reacting instead of planning. Need at least 2 weeks of buffer with pre-built assets ready to deploy.
2. **Audience Segmentation**: We treated everyone as one broad audience grouping (over-broad audience groupings). Segments must be treated distinctly.
3. **Competitor Monitoring**: Pricing monitoring must be weekly, not monthly, as competitors adjust pricing mid-campaign.

Casey: Mid-campaign budget reallocation requires 3-day lead time for platform algorithm re-learning. Don't reallocate without leadership approval.
Alex: Micro-creator influencer partnerships delivered 35% lower CPA than paid social ads. ROI exceeded expectations.

## Recurring Action Items
- Draft post-mortem, send creative brief, update audience segments, monitor competitor pricing, and update campaign tracker.
"""
    elif index == 3:
        campaign = "Back-to-School 2023"
        brand = "Horizon"
        date = "2023-09-05"
        participants = ["Riley", "Casey", "Alex", "Taylor"]
        body = f"""# Slack Thread Summary: Back-to-school campaign retrospective

**Date:** 2023-09-05
**Channel:** #marketing-horizon
**Participants:** Riley, Casey, Alex, Taylor
**Campaign Context:** Back-to-School 2023

## Thread Summary
Riley: Back-to-school is done. We left 30% of revenue on the table by not launching email earlier.
Casey: Hard agree. The 3-day delay because legal had to review copy cost us the opening weekend window (launch delays).
Alex: We had a discount code leak. Someone posted it early, attracting one-time buyers who return at 3x average rate. We need gated offers next time.
Casey: TikTok creative was a surprise hit, significantly outperforming Instagram. Recommend higher TikTok allocation.
Riley: Regarding creative fatigue refresh timing: CTR fatigues quickly (typically by day 4-5) causing CTR to drop 40-70% week-over-week. Need at least a 48-hour creative refresh buffer.

## Recurring Action Items
- Draft post-mortem with learnings, send creative brief, update segment definitions, monitor competitor pricing weekly, update campaign tracker.
"""
    else:
        return {}

    return {
        "doc_id": make_doc_id("slack_thread", index),
        "title": f"Slack Thread: {campaign} retrospective — {brand}",
        "doc_type": "slack_thread",
        "campaign": campaign,
        "brand": brand,
        "date": date,
        "participants": participants,
        "body": body,
    }


# ── Random Document Generators for Background Noise ───────────────────────────

def gen_campaign_brief_random(index: int) -> dict:
    campaign = random.choice(CAMPAIGNS)
    brand = random.choice(BRANDS)
    channels = random.sample(CHANNELS, k=random.randint(2, 4))
    audience = random.choice(AUDIENCES)
    budget = random.randint(50, 500) * 1000
    goal_roas = round(random.uniform(2.5, 6.0), 1)
    goal_cpa = random.randint(18, 65)
    date = random_date()

    body = f"""# Campaign Brief: {campaign} — {brand}

**Date:** {date}
**Author:** Marketing Strategy Team
**Campaign:** {campaign}
**Brand:** {brand}

## Objective
Drive top-of-funnel awareness and convert new-to-brand customers during the {campaign} window. Primary KPI is new customer acquisition at a target CPA of ${goal_cpa}. Secondary KPI is ROAS ≥ {goal_roas}x on paid channels.

## Target Audience
{audience}. Psychographic profile: value-seeking but brand-aware. Prior purchase data shows 68% are first-time buyers in the category. Retargeting pool is approximately 220,000 cookied visitors from the past 90 days.

## Channels & Budget Allocation
Total budget: ${budget:,}
{"".join([f"- **{ch}**: {random.randint(10, 35)}% of budget{chr(10)}" for ch in channels])}

## Creative Direction
Lead with UGC-style short-form video on {channels[0]}. Secondary assets are static carousel ads emphasizing the {random.choice(['product benefit', 'value proposition', 'social proof', 'limited-time offer'])}. All creatives must include the campaign hashtag.

## Messaging Framework
- Primary message: "{random.choice(['Performance you can feel.', 'Made for the moments that matter.', 'Less noise. More signal.', 'Built different.', 'Designed for your pace.'])}"
- Tone: {random.choice(['Aspirational but grounded', 'Energetic and direct', 'Warm and authentic', 'Bold and confident'])}
- CTA: {random.choice(['Shop Now', 'Discover More', 'Get Yours', 'Claim Your Offer', 'Start Today'])}

## Risk Flags
- Competitor {random.choice(COMPETITORS)} is running a similar promotion in the same window; monitor share of voice weekly.
- Creative fatigue risk on {channels[0]} after week 2 — have refresh assets ready.

## Success Metrics
| Metric | Target |
|--------|--------|
| New Customer CPA | ${goal_cpa} |
| ROAS | {goal_roas}x |
| Impressions | {random.randint(5, 20)}M |
| CTR | ≥ {round(random.uniform(1.2, 3.5), 1)}% |
| Email Open Rate | ≥ {random.randint(22, 38)}% |
"""
    return {
        "doc_id": make_doc_id("campaign_brief", index),
        "title": f"Campaign Brief: {campaign} — {brand}",
        "doc_type": "campaign_brief",
        "campaign": campaign,
        "brand": brand,
        "date": date,
        "channels": channels,
        "audience": audience,
        "budget_usd": budget,
        "body": body,
    }


def gen_post_mortem_random(index: int) -> dict:
    campaign = random.choice(CAMPAIGNS)
    brand = random.choice(BRANDS)
    channels = random.sample(CHANNELS, k=random.randint(2, 4))
    date = random_date()

    planned_roas = round(random.uniform(2.5, 5.0), 1)
    actual_roas = round(planned_roas + random.uniform(-1.5, 1.5), 1)
    planned_cpa = random.randint(20, 60)
    actual_cpa = int(planned_cpa * random.uniform(0.7, 1.4))
    revenue = random.randint(200, 2000) * 1000
    spend = int(revenue / actual_roas)

    top_channel = random.choice(channels)
    weak_channel = random.choice([c for c in channels if c != top_channel])
    what_worked = random.choice([
        f"UGC creatives on {top_channel} dramatically outperformed produced video (3.2x higher CTR).",
        f"Email segmentation by purchase history drove {random.randint(30, 60)}% of total revenue.",
        f"Early-bird discount landing page converted at {round(random.uniform(4, 12), 1)}%, beating our 3% benchmark.",
        f"Influencer partnership with micro-creators drove {random.randint(15, 40)}% lower CPA than brand-paid ads.",
        f"Retargeting window shortened from 30 to 7 days reduced wasted spend by {random.randint(20, 40)}%.",
    ])
    what_failed = random.choice([
        f"{weak_channel} underperformed — audience saturation hit by day 5, CTR dropped {random.randint(40, 70)}% week-over-week.",
        f"Broad match keywords on Google Ads drove unqualified traffic; should have used exact match from the start.",
        f"Late creative delivery (3 days before launch) meant no time for A/B testing. One creative ran for the entire campaign.",
        f"Messaging inconsistency between email and paid channels confused users; click-to-purchase gap was {random.randint(18, 35)}%.",
        f"Coupon code leaked to deal forums, attracting {random.randint(20, 40)}% deal-seekers who returned at 3x the average rate.",
    ])
    lesson = random.choice([
        "Never launch without at least 3 creative variants per placement.",
        "Build a 48-hour creative refresh buffer into every campaign plan.",
        "Exclude known deal-seeker segments from discount-forward campaigns.",
        "Email segmentation pays for itself — invest in clean list hygiene before Q4.",
        "Don't buy Google Ads inventory without a 7-day test budget first.",
        "Competitor spend monitoring needs to be weekly, not monthly — they moved first.",
    ])

    body = f"""# Post-Mortem: {campaign} — {brand}

**Date:** {date}
**Author:** Performance Marketing Team
**Campaign:** {campaign}

## Executive Summary
The {campaign} campaign ran from [start] to [end] with a total spend of ${spend:,}. We generated ${revenue:,} in attributable revenue, achieving a ROAS of {actual_roas}x against a target of {planned_roas}x. New customer CPA came in at ${actual_cpa} vs. a planned ${planned_cpa}.

## What Worked
{what_worked}

Details: {top_channel} drove {random.randint(35, 60)}% of total campaign revenue at the lowest CPA of any channel. The creative format that resonated most was {random.choice(['short-form video under 15 seconds', 'carousel showing before/after', 'static with bold headline + testimonial', 'story format with swipe-up CTA'])}.

## What Didn't Work
{what_failed}

This cost us an estimated ${random.randint(15, 80) * 1000:,} in wasted spend and contributed to the CPA miss.

## Key Learnings
1. {lesson}
2. Budget reallocation mid-campaign is possible but requires a 3-day lead time for platform algorithm re-learning.
3. {random.choice(['Retargeting lists must be refreshed weekly during a campaign, not monthly.', 'Attribution model disagreement between channels inflated our ROAS estimate by ~15%.', 'Audience overlap between channels caused frequency spikes — add exclusion lists.'])}

## Recommendations for Next Year
- Increase {top_channel} allocation by {random.randint(10, 25)}% — highest efficiency channel.
- Reduce {weak_channel} to a test budget only until creative format is validated.
- Build competitor monitoring into weekly ops cadence, not a one-time pre-launch check.
- Create a "creative vault" with evergreen assets ready to deploy when campaign creative fatigues.

## Financial Summary
| Metric | Planned | Actual |
|--------|---------|--------|
| Total Spend | ${spend:,} | ${int(spend * random.uniform(0.95, 1.1)):,} |
| Revenue | — | ${revenue:,} |
| ROAS | {planned_roas}x | {actual_roas}x |
| New Customer CPA | ${planned_cpa} | ${actual_cpa} |
"""
    return {
        "doc_id": make_doc_id("post_mortem", index),
        "title": f"Post-Mortem: {campaign} — {brand}",
        "doc_type": "post_mortem",
        "campaign": campaign,
        "brand": brand,
        "date": date,
        "channels": channels,
        "actual_roas": actual_roas,
        "actual_cpa": actual_cpa,
        "body": body,
    }


def gen_customer_research_random(index: int) -> dict:
    persona = random.choice(PERSONAS)
    brand = random.choice(BRANDS)
    date = random_date()
    n_respondents = random.randint(250, 1200)
    nps = random.randint(22, 72)
    top_driver = random.choice(["product quality", "brand trust", "price-value ratio", "sustainability", "ease of purchase"])
    churn_reason = random.choice(["price sensitivity", "found a better alternative", "product didn't meet expectations", "shipping issues", "poor customer service"])

    body = f"""# Customer Research Report: {persona['name']} Segment — {brand}

**Date:** {date}
**Research Method:** Online survey (n={n_respondents}) + 12 customer interviews
**Brand:** {brand}
**Segment:** {persona['name']} (Age {persona['age']}, HHI {persona['income']})

## Executive Summary
This research explores the attitudes, purchase drivers, and friction points of the {persona['name']} segment for {brand}. The segment represents approximately {random.randint(22, 38)}% of our customer base but {random.randint(35, 55)}% of revenue. NPS for this segment is {nps}.

## Purchase Drivers (Top 3)
1. **{top_driver.title()}** — cited by {random.randint(52, 78)}% of respondents as a primary purchase trigger.
2. **{random.choice(['word of mouth', 'social media discovery', 'influencer recommendation', 'past brand experience'])}** — {random.randint(38, 55)}% said this was their first touchpoint.
3. **{random.choice(['product reviews on site', 'free shipping threshold', 'loyalty program', 'easy returns policy'])}** — {random.randint(30, 50)}% cited this as a final conversion factor.

## Churn & Lapsed Buyer Analysis
The primary reason this segment lapses is **{churn_reason}** ({random.randint(30, 50)}% of lapsed buyers). Secondary reason: **{random.choice(['lack of new products', 'email fatigue', 'found better pricing elsewhere', 'changed life circumstances'])}** ({random.randint(18, 32)}%).

A reactivation email campaign tested in Q{random.randint(1, 4)} showed a {round(random.uniform(8, 22), 1)}% reactivation rate when paired with a personalized offer based on past purchase history.

## Brand Perception
When asked to describe {brand} in three words, the most common responses were:
- "{random.choice(['reliable', 'innovative', 'premium', 'trustworthy', 'modern'])}" — {random.randint(35, 60)}%
- "{random.choice(['stylish', 'practical', 'smart', 'clean', 'bold'])}" — {random.randint(25, 45)}%
- "{random.choice(['expensive', 'worth it', 'fast', 'authentic', 'accessible'])}" — {random.randint(20, 40)}%

## Messaging Resonance Test
We tested 6 messaging concepts. Top performer: **"{random.choice(['Built for people who expect more.', 'The smarter choice for what matters.', 'Real results. No compromise.', 'Designed around your life.'])}"** with a {random.randint(55, 80)}% positive response rate.

Worst performer: **"{random.choice(['Luxury for less.', 'Join the movement.', 'Because you deserve it.'])}"** — felt inauthentic to this segment.

## Persona Values
This segment values: **{persona['values']}**. Communications that lead with these values outperform category-generic messaging by {random.randint(25, 55)}% on click-through in our A/B tests.

## Recommendations
1. Lean into {top_driver} messaging in all {persona['name']} segment targeting.
2. Re-examine pricing architecture — {random.randint(28, 42)}% said they would pay {random.randint(10, 25)}% more if quality signaling was clearer.
3. Reactivation program should be automated and triggered at day 90 of no purchase.
"""
    return {
        "doc_id": make_doc_id("customer_research", index),
        "title": f"Customer Research: {persona['name']} Segment — {brand}",
        "doc_type": "customer_research",
        "brand": brand,
        "persona": persona["name"],
        "date": date,
        "body": body,
    }


def gen_win_loss_random(index: int) -> dict:
    competitor = random.choice(COMPETITORS)
    brand = random.choice(BRANDS)
    date = random_date()
    won = random.choice([True, True, False])
    deal_size = random.randint(20, 500) * 1000

    if won:
        reason = random.choice([
            f"Prospect cited {brand}'s stronger analytics dashboard and onboarding support vs. {competitor}.",
            f"Price was 15% higher than {competitor} but the prospect valued the SLA guarantees.",
            f"{brand}'s case study library from similar companies was the deciding factor — {competitor} had none.",
            f"Faster implementation timeline (6 weeks vs. {competitor}'s 14) won the deal.",
        ])
        loss_lesson = "N/A — won this deal."
    else:
        reason = random.choice([
            f"{competitor} matched our price and offered an extra year of premium support.",
            f"Prospect had an existing relationship with {competitor} — we couldn't displace.",
            f"Our demo environment had performance issues; {competitor}'s was flawless.",
            f"We lacked a specific compliance certification that {competitor} holds.",
        ])
        loss_lesson = random.choice([
            "We need a competitive battle card specifically for head-to-head situations with this competitor.",
            "Demo environment reliability must be treated as a sales-critical infrastructure item.",
            "Compliance gap is blocking deals in regulated verticals — prioritize certification.",
            "Relationship-first accounts require multi-threading 6+ months before evaluation begins.",
        ])

    body = f"""# Win/Loss Note: {brand} vs. {competitor}

**Date:** {date}
**Outcome:** {"WIN ✅" if won else "LOSS ❌"}
**Deal Size:** ${deal_size:,}
**Analyst:** Sales Strategy Team

## Situation
Competitive evaluation between {brand} and {competitor} for a {random.choice(['mid-market retailer', 'DTC brand', 'CPG company', 'regional grocery chain', 'subscription box company'])} account. Evaluation ran over {random.randint(3, 12)} weeks.

## Why We {"Won" if won else "Lost"}
{reason}

## Decision Criteria (Ranked by Prospect)
1. {random.choice(['Integration capabilities', 'Price', 'Support quality', 'Feature set', 'Implementation speed'])}
2. {random.choice(['Ease of use', 'Reporting depth', 'Scalability', 'Brand reputation', 'Customer references'])}
3. {random.choice(['Contract flexibility', 'Onboarding quality', 'Security posture', 'Innovation roadmap'])}

## Competitor Intel
{competitor} is aggressively discounting to win new logos — we saw {random.randint(15, 35)}% off list price in this deal. Their new {random.choice(['mobile app', 'AI feature', 'integration', 'dashboard'])} appears to be resonating with technical buyers.

## Lesson / Action Item
{loss_lesson}

## Follow-Up
{"Closed. Moving to onboarding." if won else f"Prospect open to revisiting in Q{random.randint(1, 4)} {random.randint(2024, 2025)}. Schedule a check-in."}
"""
    return {
        "doc_id": make_doc_id("win_loss", index),
        "title": f"Win/Loss: {brand} vs. {competitor} ({'Win' if won else 'Loss'})",
        "doc_type": "win_loss",
        "brand": brand,
        "competitor": competitor,
        "date": date,
        "won": won,
        "deal_size_usd": deal_size,
        "body": body,
    }


def gen_slack_thread_random(index: int) -> dict:
    campaign = random.choice(CAMPAIGNS)
    brand = random.choice(BRANDS)
    date = random_date()
    names = ["Alex", "Jamie", "Morgan", "Taylor", "Jordan", "Casey", "Riley", "Drew"]
    participants = random.sample(names, k=random.randint(3, 5))

    topics = [
        (f"Quick sync on {campaign} performance",
         f"{participants[0]}: Hey team, just pulled the numbers for {campaign}. ROAS is tracking at {round(random.uniform(2, 5), 1)}x which is {'above' if random.random() > 0.5 else 'below'} target.\n{participants[1]}: Yeah I saw that. Email is carrying us — paid social has been weak since day 4.\n{participants[2]}: Agreed. I think we need to reallocate some budget from {random.choice(CHANNELS)} to {random.choice(CHANNELS)}.\n{participants[0]}: Let's not do that without checking with leadership. But I can pull the channel breakdown.\n{participants[3] if len(participants) > 3 else participants[0]}: The creative fatigue issue is real. We need refresh assets by EOW.\n{participants[1]}: I'll flag it to design. They'll need 48 hours minimum.\n{participants[0]}: Got it. I'll send a brief today. Everyone aligned?"),
        (f"Lessons from {campaign} — what should we do differently?",
         f"{participants[0]}: Okay wrapping up {campaign}. What are the top 3 learnings everyone wants documented?\n{participants[1]}: Creative refresh timeline. We were reacting instead of planning. Need at least 2 weeks of buffer.\n{participants[2]}: Audience segmentation. We treated {random.choice(AUDIENCES)} as one blob. They're not.\n{participants[0]}: Strong. What else?\n{participants[3] if len(participants) > 3 else participants[1]}: Competitor monitoring. {random.choice(COMPETITORS)} dropped their price on day 3 and we didn't know for a week.\n{participants[0]}: That one stings. Okay I'll write up the post-mortem with these as the anchors. Anything else?\n{participants[2]}: The influencer collab was actually really good. Higher ROI than expected. We should double down next time."),
        (f"Back-to-school campaign retrospective",
         f"{participants[0]}: Back-to-school is done. Here's my hot take: we left {random.randint(15, 30)}% of revenue on the table by not launching email earlier.\n{participants[1]}: Hard agree. The {random.randint(3, 7)}-day delay because legal had to review copy cost us the opening weekend window.\n{participants[2]}: We also had the discount code leak issue. Someone in the thread posted it early.\n{participants[0]}: Yeah that attracted a bunch of one-time buyers who will never return. We need a gated offer approach next time.\n{participants[3] if len(participants) > 3 else participants[1]}: The TikTok creative was a surprise hit though. Way outperformed our Instagram.\n{participants[0]}: Yeah we need to rethink our channel mix for next year. TikTok budget needs to go up."),
    ]

    topic_title, thread_body = random.choice(topics)

    body = f"""# Slack Thread Summary: {topic_title}

**Date:** {date}
**Channel:** #marketing-{brand.lower()}
**Participants:** {', '.join(participants)}
**Campaign Context:** {campaign}

## Thread Summary

{thread_body}

## Key Decisions Made
- {random.choice(['Budget reallocation to be reviewed by leadership before actioning.', 'Creative refresh brief to go out same day.', 'Post-mortem document to be written capturing top 3 learnings.', 'Influencer collab to be expanded in next campaign cycle.', 'Competitor monitoring to move to weekly cadence.'])}

## Action Items
| Owner | Action | Due |
|-------|--------|-----|
| {participants[0]} | {random.choice(['Draft post-mortem', 'Pull channel breakdown', 'Send creative brief', 'Update campaign tracker'])} | EOW |
| {participants[1]} | {random.choice(['Flag to design team', 'Update audience segments', 'Review offer mechanics', 'Monitor competitor pricing'])} | {random.randint(2, 5)}d |
"""
    return {
        "doc_id": make_doc_id("slack_thread", index),
        "title": f"Slack Thread: {topic_title} — {brand}",
        "doc_type": "slack_thread",
        "campaign": campaign,
        "brand": brand,
        "date": date,
        "participants": participants,
        "body": body,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)

    total = 0

    # 1. Campaign briefs (110)
    for i in range(1, 111):
        if i <= 6:
            doc = get_low_index_campaign_brief(i)
        else:
            doc = gen_campaign_brief_random(i)
        out_path = os.path.join(OUTPUT_DIR, f"{doc['doc_id']}.json")
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        total += 1

    # 2. Post-mortems (110)
    for i in range(1, 111):
        if i <= 7:
            doc = get_low_index_post_mortem(i)
        else:
            doc = gen_post_mortem_random(i)
        out_path = os.path.join(OUTPUT_DIR, f"{doc['doc_id']}.json")
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        total += 1

    # 3. Customer research (100)
    for i in range(1, 101):
        if i <= 6:
            doc = get_low_index_customer_research(i)
        else:
            doc = gen_customer_research_random(i)
        out_path = os.path.join(OUTPUT_DIR, f"{doc['doc_id']}.json")
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        total += 1

    # 4. Win/loss notes (100)
    for i in range(1, 101):
        if i <= 7:
            doc = get_low_index_win_loss(i)
        else:
            doc = gen_win_loss_random(i)
        out_path = os.path.join(OUTPUT_DIR, f"{doc['doc_id']}.json")
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        total += 1

    # 5. Slack threads (100)
    for i in range(1, 101):
        if i <= 3:
            doc = get_low_index_slack_thread(i)
        else:
            doc = gen_slack_thread_random(i)
        out_path = os.path.join(OUTPUT_DIR, f"{doc['doc_id']}.json")
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        total += 1

    print(f"✅ Generated {total} documents in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
