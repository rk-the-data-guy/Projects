import json
import random
import os

CATEGORIES = ["Electronics", "Home & Kitchen", "Apparel", "Sports & Outdoors"]
COMPETITORS = ["Amazon", "Walmart", "Target", "BestBuy"]

def generate_sku(index):
    category = random.choice(CATEGORIES)
    cogs = round(random.uniform(10.0, 200.0), 2)
    # Price is usually cogs + 20% to 100% markup
    current_price = round(cogs * random.uniform(1.2, 2.0), 2)
    inventory = random.randint(0, 500)
    sales_velocity = round(random.uniform(0.5, 50.0), 1)
    elasticity = round(random.uniform(-2.5, -0.5), 2)

    num_competitors = random.randint(1, 3)
    competitor_pricing = []
    selected_comps = random.sample(COMPETITORS, num_competitors)
    
    for comp in selected_comps:
        # Competitor price is around our current price +/- 15%
        comp_price = round(current_price * random.uniform(0.85, 1.15), 2)
        in_stock = random.random() > 0.1 # 90% chance in stock
        competitor_pricing.append({
            "competitor_name": comp,
            "price": comp_price,
            "in_stock": in_stock
        })

    return {
        "sku_id": f"SKU-{index:04d}",
        "name": f"{category} Item {index}",
        "category": category,
        "cogs": cogs,
        "current_price": current_price,
        "inventory_level": inventory,
        "sales_velocity_weekly": sales_velocity,
        "elasticity_estimate": elasticity,
        "competitor_pricing": competitor_pricing
    }

def main():
    skus = [generate_sku(i) for i in range(1, 201)]
    
    os.makedirs("data", exist_ok=True)
    with open("data/mock_skus.json", "w") as f:
        json.dump(skus, f, indent=2)
        
    print(f"Synthesized {len(skus)} mock SKUs and saved to data/mock_skus.json")

if __name__ == "__main__":
    main()
