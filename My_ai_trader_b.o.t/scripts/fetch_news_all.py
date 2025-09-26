# scripts/fetch_news_all.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json, os
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from user_data.data.news_predictor import get_news_prediction, MONITORED

out = {}
for sym in MONITORED.keys():
    try:
        pred = get_news_prediction(sym)
        out[sym] = pred
    except Exception as e:
        out[sym] = {"error": str(e)}

# write results
os.makedirs("logs", exist_ok=True)
with open("logs/predictions.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("Wrote logs/predictions.json")
