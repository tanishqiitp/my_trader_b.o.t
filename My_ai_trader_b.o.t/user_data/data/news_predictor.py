# user_data/data/news_predictor.py
"""
News-based directional predictor for commodities, crypto, and NIFTY50 stocks.

Returns:
    get_news_prediction(symbol) -> dict {
        "symbol": "RELIANCE",
        "score": 0.23,             # combined numeric score (sentiment + event boost)
        "prediction": 1,          # 1 => predict up next day, -1 => predict down next day, 0 => neutral
        "reasons": ["OPEC cut keywords found", "VADER positive 0.18"],
        "headlines": [... up to 10 headlines ...]
    }
"""


import os, time, json, logging, re
from typing import List, Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import feedparser
from .news_sentiment import fetch_headlines_newsapi
from .news_sentiment import fetch_headlines_newsapi, fetch_headlines_rss, score_texts, _read_cache, _write_cache, CACHE_TTL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

analyzer = SentimentIntensityAnalyzer()

# Configure monitored sets
# NOTE: adjust aliases/names to match common news usage
import json
import os

ASSETS_FILE = os.path.join(os.path.dirname(__file__), "monitored_assets.json")

# Load monitored assets dynamically
with open(ASSETS_FILE, "r", encoding="utf-8") as f:
    MONITORED = json.load(f)

def get_monitored_assets():
    """Return list of available assets dynamically from JSON."""
    return list(MONITORED.keys())


# Event keywords and their directional effect per asset group
# value: +1 for bullish, -1 for bearish, 0 for neutral
EVENT_KEYWORDS = [
    # For crude/oil: geopolitical events that push price up
    (r"\bopec\b|\bproduction cut\b|\bcut production\b|\bexport ban\b|\boil embargo\b|\boil sanctions\b|\boil supply cut\b", {"CRUDE": 1}),
    (r"\bsanction\b|\bembargo\b|\boil supply cut\b|\bblockade\b|\bwar\b|\binvasion\b", {"CRUDE": 1, "GOLD": 1}),  # war/sanctions -> oil/gold up
    # Geopolitical easing -> crude down
    (r"\bceasefire\b|\bpeace accord\b|\bdeal reached\b|\bproduction increase\b|\bboost production\b", {"CRUDE": -1}),
    # For gold: flight-to-safety keywords
    (r"\bmarket panic\b|\brecession\b|\binflation spike\b|\bflight to safety\b|\bbank run\b", {"GOLD": 1}),
    # For stocks: earnings/contract news
    (r"\b(q[1-4] results|quarterly results|beats estimates|beats expectations|missed estimates|disappointing results|beat expectations)\b",
     {"STOCK_POSITIVE": 1, "STOCK_NEGATIVE": -1}),
    # Company wins major contract or large order -> positive
    (r"\bwon contract\b|\bsecured contract\b|\bawarded contract\b|\bwon deal\b|\bstrategic partnership\b|\bbagged\b", {"STOCK_POSITIVE": 1}),
    # Mergers, regulatory fines -> negative for companies
    (r"\bfine\b|\bpenalty\b|\binvestigation\b|\brecall\b|\blawsuit\b", {"STOCK_NEGATIVE": -1}),
]

# Cache storage path (re-using news_cache in the other module)
CACHE_FILE = os.path.join(os.path.dirname(__file__), "news_cache.json")

def _get_headlines_for_query(query: str, use_newsapi=True, max_items=25) -> List[str]:
    """Try NewsAPI first if key present, otherwise fallback to Google News RSS."""
    api_key = os.environ.get("NEWSAPI_KEY")
    headlines = []
    if use_newsapi and api_key:
        try:
            headlines = fetch_headlines_newsapi(query, api_key, page_size=max_items)
        except Exception as e:
            logger.warning("NewsAPI fetch failed for '%s': %s", query, e)
            headlines = []
    if not headlines:
        try:
            headlines = fetch_headlines_rss(query, max_items)
        except Exception as e:
            logger.warning("RSS fetch failed for '%s': %s", query, e)
            headlines = []
    return headlines

def detect_events_in_texts(texts: List[str]) -> Dict[str, int]:
    """
    Returns a dict of detected event signals aggregated:
      'CRUDE': score, 'GOLD': score, 'STOCK_POSITIVE': count, 'STOCK_NEGATIVE': count
    """
    agg = {}
    txt = " ".join(texts).lower()
    for pattern, effects in EVENT_KEYWORDS:
        if re.search(pattern, txt, re.IGNORECASE):
            for k, v in effects.items():
                agg[k] = agg.get(k, 0) + v
    return agg

def compute_combined_score(sentiment_score: float, event_agg: Dict[str, int], symbol_key: str) -> float:
    """
    Combine VADER sentiment (float in [-1,1]) and event signals into a single score.
    We weigh event signals higher for commodities (fast-moving).
    """
    base = sentiment_score
    event_score = 0.0
    # weight rules
    commodity_weight = 0.6
    stock_weight = 0.4
    # CRUDE/GOLD direct events:
    if symbol_key == "CRUDE":
        event_score += celebrity_event_value(event_agg.get("CRUDE", 0)) * commodity_weight
        event_score += celebrity_event_value(event_agg.get("GOLD", 0)) * 0.2  # cross-effect
    elif symbol_key == "GOLD":
        event_score += celebrity_event_value(event_agg.get("GOLD", 0)) * commodity_weight
        event_score += celebrity_event_value(event_agg.get("CRUDE", 0)) * 0.2
    else:
        # for stocks: evaluate STOCK_POSITIVE/NEGATIVE
        sp = event_agg.get("STOCK_POSITIVE", 0)
        sn = event_agg.get("STOCK_NEGATIVE", 0)
        event_score += (sp - sn) * stock_weight

    # final weighted sum
    # normalize sentiment contribution to moderate effect
    return 0.6 * base + 0.4 * event_score

def celebrity_event_value(v: int) -> float:
    """
    Convert event count to normalized value, capped.
    """
    if v == 0:
        return 0.0
    # simple scaling: each event occurrence contributes 0.5
    return max(-2.0, min(2.0, v * 0.5))

def get_news_prediction(symbol_key: str, aliases: List[str] = None, use_newsapi=True, max_headlines=25) -> Dict[str, Any]:
    """
    symbol_key: one of MONITORED keys e.g., 'BTC', 'CRUDE', 'GOLD', 'RELIANCE'...
    Returns dict with numeric score and discrete prediction.
    """
    symbol_key = symbol_key.upper()
    if symbol_key not in MONITORED:
        raise KeyError(f"{symbol_key} not in monitored list")

    aliases = aliases or MONITORED[symbol_key].get("aliases", [symbol_key])
    query = " OR ".join(aliases)

    # Caching: reuse shared CACHE_FILE (small TTL)
    try:
        cache = _read_cache()
    except Exception:
        cache = {}

    cache_entry = cache.get(symbol_key)
    if cache_entry and (time.time() - cache_entry.get("ts", 0) < CACHE_TTL):
        logger.debug("Using cached prediction for %s", symbol_key)
        return cache_entry

    headlines = _get_headlines_for_query(query, use_newsapi=use_newsapi, max_items=max_headlines)
    sentiment = 0.0
    if headlines:
        sentiment = score_texts(headlines)  # using VADER wrapper from news_sentiment.py
    event_agg = detect_events_in_texts(headlines)

    score = compute_combined_score(sentiment, event_agg, symbol_key)
    # thresholds: tune via backtesting
    threshold = 0.15
    prediction = 0
    if score >= threshold:
        prediction = 1
    elif score <= -threshold:
        prediction = -1
    reasons = []
    if event_agg:
        reasons.append(f"events:{event_agg}")
    reasons.append(f"vader:{sentiment:.3f}")
    result = {
        "symbol": symbol_key,
        "score": round(score, 4),
        "prediction": prediction,
        "reasons": reasons,
        "headlines": headlines[:10],
        "ts": int(time.time())
    }

    cache[symbol_key] = result
    try:
        _write_cache(cache)
    except Exception:
        pass
    return result

# Simple CLI to fetch for all monitored symbols
if __name__ == "__main__":
    import sys
    keys = sys.argv[1:] or list(MONITORED.keys())
    for k in keys:
        print(get_news_prediction(k))
