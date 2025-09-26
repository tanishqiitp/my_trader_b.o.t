# scripts/test_system.py
"""
Quick sanity test for news_sentiment + news_predictor integration.

Runs:
1. Import checks
2. BTC single prediction
3. Multi-asset fetch (BTC, CRUDE, GOLD, RELIANCE if in MONITORED)
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from user_data.data.news_predictor import get_monitored_assets

print("✅ Monitored Assets Loaded:", get_monitored_assets())


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("=== Step 1: Import Check ===")
try:
    from user_data.data import news_sentiment, news_predictor
    print("✅ user_data.data imports successful!")
except Exception as e:
    print("❌ Import failed:", e)
    sys.exit(1)

print("Functions in news_sentiment:", [f for f in dir(news_sentiment) if not f.startswith("__")])
print("Functions in news_predictor:", [f for f in dir(news_predictor) if not f.startswith("__")])

print("\n=== Step 2: BTC Prediction Test ===")
try:
    result = news_predictor.get_news_prediction("BTC")
    print(json.dumps(result, indent=2))
except Exception as e:
    print("❌ BTC prediction failed:", e)

print("\n=== Step 3: Multi-Asset Fetch Test ===")
from user_data.data.news_predictor import MONITORED, get_news_prediction
out = {}
for sym in ["BTC", "CRUDE", "GOLD"]:
    if sym in MONITORED:
        try:
            out[sym] = get_news_prediction(sym)
        except Exception as e:
            out[sym] = {"error": str(e)}

# Write results to logs
os.makedirs("logs", exist_ok=True)
with open("logs/test_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"✅ Multi-asset results written to logs/test_results.json")
