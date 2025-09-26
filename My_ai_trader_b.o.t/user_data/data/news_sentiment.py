# user_data/data/news_sentiment.py
"""
Lightweight news-fetch + sentiment caching.
Uses NEWSAPI if NEWSAPI_KEY is set in env, otherwise falls back to Google News RSS.
Scoring: VADER (fast + small). Cache TTL default: 15 minutes.
"""

import os
import time
import json
import requests
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import logging

# set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make cache file path relative to this module file (robust)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "news_cache.json")
CACHE_TTL = 15 * 60   # seconds

analyzer = SentimentIntensityAnalyzer()

def _read_cache() -> dict:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_cache(d: dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def fetch_headlines_newsapi(query: str, api_key: str, page_size=20) -> List[str]:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    return [((a.get("title") or "") + ". " + (a.get("description") or "")) for a in data.get("articles", [])]

def fetch_headlines_rss(query: str, max_items=20) -> List[str]:
    rss_url = f"https://news.google.com/rss/search?q={requests.utils.requote_uri(query)}"
    feed = feedparser.parse(rss_url)
    return [getattr(entry, "title", "") for entry in feed.entries[:max_items]]

def score_texts(texts: List[str]) -> float:
    if not texts:
        return 0.0
    scores = [analyzer.polarity_scores(t)["compound"] for t in texts if t]
    return sum(scores) / len(scores) if scores else 0.0

def get_sentiment(symbol: str, aliases: List[str] = None, use_newsapi=True) -> float:
    symbol = symbol.upper()
    cache = _read_cache()
    entry = cache.get(symbol)
    if entry and (time.time() - entry.get("ts", 0) < CACHE_TTL):
        logger.debug("Using cached sentiment for %s", symbol)
        return entry.get("score", 0.0)

    query_parts = [symbol] + (aliases or [])
    query = " OR ".join(query_parts)

    headlines = []
    api_key = os.environ.get("NEWSAPI_KEY")
    if use_newsapi and api_key:
        try:
            headlines = fetch_headlines_newsapi(query, api_key)
        except Exception as e:
            logger.warning("NewsAPI fetch failed: %s", e)
            headlines = []

    if not headlines:
        try:
            headlines = fetch_headlines_rss(query)
        except Exception as e:
            logger.warning("RSS fetch failed: %s", e)
            headlines = []

    score = score_texts(headlines)
    cache[symbol] = {"ts": int(time.time()), "score": score, "headlines": headlines[:10]}
    _write_cache(cache)
    logger.info("Fetched sentiment for %s -> %.3f (headlines=%d)", symbol, score, len(headlines))
    return score

# CLI quick test if run directly
if __name__ == "__main__":
    import sys
    syms = sys.argv[1:] or ["BTC", "ETH"]
    for s in syms:
        print(s, get_sentiment(s))
