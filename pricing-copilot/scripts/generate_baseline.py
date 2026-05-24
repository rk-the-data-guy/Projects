import json

def determine_baseline(sku):
    inventory = sku["inventory_level"]
    sales = sku["sales_velocity_weekly"]
    price = sku["current_price"]
    comps = sku["competitor_pricing"]
    
    if inventory > 300 and sales < 10:
        return "DECREASE"
        
    if inventory < 20 and sales > 30:
        return "INCREASE"
        
    if comps:
        avg_comp_price = sum(c["price"] for c in comps) / len(comps)
        if price > avg_comp_price * 1.15:
            return "DECREASE"
        if price < avg_comp_price * 0.85:
            return "INCREASE"
            
    return "HOLD"

def main():
    with open("data/mock_skus.json", "r") as f:
        skus = json.load(f)
        
    baselines = {}
    for sku in skus:
        baselines[sku["sku_id"]] = determine_baseline(sku)
        
    with open("data/baseline.json", "w") as f:
        json.dump(baselines, f, indent=2)
        
    print(f"Generated {len(baselines)} baseline decisions.")

if __name__ == "__main__":
    main()
