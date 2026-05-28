"""Hybrid stock recommendation engine driven by market news.

The public entry points are:
    get_news_prediction(symbol_key) -> one stock/asset signal
    analyze_watchlist(watchlist) -> grouped TO_BUY/HOLD/SELL dashboard data

Rules and VADER sentiment are always available. A trained ML classifier and a
FinBERT DL sentiment backend can be layered in when their dependencies/models
are installed.
"""

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .news_sentiment import (
    CACHE_TTL,
    _read_cache,
    _write_cache,
    fetch_headlines_newsapi,
    fetch_headlines_rss,
    score_texts,
)
from .ml_signal import predict_news_action_ml, score_news_with_finbert

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ASSETS_FILE = os.path.join(os.path.dirname(__file__), "monitored_assets.json")
ENGINE_VERSION = "hybrid-news-v3"

ACTION_TO_BUY = "TO_BUY"
ACTION_HOLD = "HOLD"
ACTION_SELL = "SELL"
ACTIONS = (ACTION_TO_BUY, ACTION_HOLD, ACTION_SELL)
MODEL_MODES = {"rules", "ml", "dl", "hybrid"}

analyzer = SentimentIntensityAnalyzer()


def load_monitored_assets() -> Dict[str, Dict[str, Any]]:
    with open(ASSETS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


MONITORED = load_monitored_assets()


def refresh_monitored_assets() -> Dict[str, Dict[str, Any]]:
    """Reload monitored assets after dashboard/API watchlist edits."""
    global MONITORED
    MONITORED = load_monitored_assets()
    return MONITORED


def get_monitored_assets() -> List[str]:
    return list(MONITORED.keys())


# Company-specific and broad market events. These are intentionally explicit so
# users can understand why a stock moved into TO_BUY/HOLD/SELL.
BULLISH_PATTERNS = [
    r"\bbeats estimates\b",
    r"\bbeats expectations\b",
    r"\brecord profit\b",
    r"\bprofit jumps\b",
    r"\brevenue rises\b",
    r"\bstrong demand\b",
    r"\bupgrade[sd]?\b",
    r"\btarget price raised\b",
    r"\bwon (a )?(large |major |multi[- ]year )?contract\b",
    r"\bsecured (a )?(large |major |multi[- ]year )?(order|contract|deal)\b",
    r"\bbagged (a )?(large |major )?(order|contract|deal)\b",
    r"\bstrategic partnership\b",
    r"\bshare buyback\b",
    r"\bdividend\b",
    r"\bexpansion\b",
    r"\bnew order\b",
    r"\bapproval\b",
]

BEARISH_PATTERNS = [
    r"\bmisses estimates\b",
    r"\bmissed estimates\b",
    r"\bprofit falls\b",
    r"\bloss widens\b",
    r"\brevenue declines\b",
    r"\bdowngrade[sd]?\b",
    r"\btarget price cut\b",
    r"\bfine\b",
    r"\bpenalty\b",
    r"\binvestigation\b",
    r"\blawsuit\b",
    r"\bdefault\b",
    r"\bfraud\b",
    r"\bregulatory action\b",
    r"\bweak demand\b",
    r"\bstrike\b",
    r"\bresigns\b",
    r"\brecall\b",
]


def _asset_config(symbol_key: str, assets: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    assets = assets or MONITORED
    key = symbol_key.upper().strip()
    if key not in assets:
        raise KeyError(f"{key} is not in the monitored list")
    return assets[key]


def _query_for_asset(symbol_key: str, config: Dict[str, Any]) -> str:
    aliases = config.get("aliases") or []
    name = config.get("name")
    symbol = config.get("symbol")
    query_parts = [symbol_key]
    if name:
        query_parts.append(name)
    if symbol:
        query_parts.append(str(symbol).replace(".NS", "").replace(".BO", ""))
    query_parts.extend(aliases)

    seen = set()
    cleaned = []
    for part in query_parts:
        value = str(part).strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            cleaned.append(value)
    return " OR ".join(cleaned)


def _get_headlines_for_query(query: str, use_newsapi: bool = True, max_items: int = 25) -> List[str]:
    """Try NewsAPI first when configured, otherwise fallback to Google News RSS."""
    api_key = os.environ.get("NEWSAPI_KEY")
    headlines: List[str] = []
    if use_newsapi and api_key:
        try:
            headlines = fetch_headlines_newsapi(query, api_key, page_size=max_items)
        except Exception as exc:
            logger.warning("NewsAPI fetch failed for '%s': %s", query, exc)

    if not headlines:
        try:
            headlines = fetch_headlines_rss(query, max_items)
        except Exception as exc:
            logger.warning("RSS fetch failed for '%s': %s", query, exc)
    return headlines


def _pattern_hits(text: str, patterns: List[str]) -> List[str]:
    hits = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append(pattern)
    return hits


def detect_stock_events(headlines: List[str]) -> Dict[str, Any]:
    joined = " ".join(headlines).lower()
    bullish_hits = _pattern_hits(joined, BULLISH_PATTERNS)
    bearish_hits = _pattern_hits(joined, BEARISH_PATTERNS)
    return {
        "bullish_count": len(bullish_hits),
        "bearish_count": len(bearish_hits),
        "bullish_patterns": bullish_hits[:5],
        "bearish_patterns": bearish_hits[:5],
    }


def compute_news_score(sentiment_score: float, events: Dict[str, Any]) -> float:
    event_balance = events.get("bullish_count", 0) - events.get("bearish_count", 0)
    event_score = max(-1.0, min(1.0, event_balance * 0.18))
    return round((0.65 * sentiment_score) + (0.35 * event_score), 4)


def action_from_score(score: float, headlines_count: int) -> str:
    if headlines_count == 0:
        return ACTION_HOLD
    if score >= 0.12:
        return ACTION_TO_BUY
    if score <= -0.12:
        return ACTION_SELL
    return ACTION_HOLD


def confidence_from_score(score: float, headlines_count: int) -> int:
    coverage_boost = min(20, headlines_count * 2)
    confidence = int(min(95, 45 + abs(score) * 100 + coverage_boost))
    if headlines_count == 0:
        return 20
    return max(25, confidence)


def _reason_summary(action: str, score: float, sentiment: float, events: Dict[str, Any]) -> List[str]:
    reasons = [f"news score {score:+.3f}", f"sentiment {sentiment:+.3f}"]
    if events.get("bullish_count"):
        reasons.append(f"{events['bullish_count']} bullish news event(s)")
    if events.get("bearish_count"):
        reasons.append(f"{events['bearish_count']} bearish news event(s)")
    if action == ACTION_HOLD and not events.get("bullish_count") and not events.get("bearish_count"):
        reasons.append("no strong event trigger")
    return reasons


def _action_to_numeric(action: str, confidence: int = 100) -> float:
    direction = {ACTION_TO_BUY: 1.0, ACTION_HOLD: 0.0, ACTION_SELL: -1.0}.get(action, 0.0)
    return direction * max(0.0, min(1.0, confidence / 100))


def _cache_key(symbol_key: str, query: str, model_mode: str, enable_finbert: bool) -> str:
    query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    return f"{ENGINE_VERSION}:{symbol_key.upper()}:{model_mode}:{int(enable_finbert)}:{query_hash}"


def _should_use_finbert(model_mode: str, enable_finbert: Optional[bool]) -> bool:
    if enable_finbert is not None:
        return enable_finbert
    if model_mode == "dl":
        return True
    return os.environ.get("NEWS_ENABLE_FINBERT", "").lower() in {"1", "true", "yes", "on"}


def get_news_prediction(
    symbol_key: str,
    aliases: Optional[List[str]] = None,
    use_newsapi: bool = True,
    max_headlines: int = 25,
    assets: Optional[Dict[str, Dict[str, Any]]] = None,
    model_mode: str = "hybrid",
    enable_finbert: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return a news-driven signal for one monitored stock/asset."""
    assets = assets or MONITORED
    symbol_key = symbol_key.upper().strip()
    model_mode = model_mode.lower().strip()
    if model_mode not in MODEL_MODES:
        raise ValueError(f"Unknown model_mode '{model_mode}'. Use one of: {sorted(MODEL_MODES)}")

    config = _asset_config(symbol_key, assets)
    effective_config = dict(config)
    if aliases is not None:
        effective_config["aliases"] = aliases
    query = _query_for_asset(symbol_key, effective_config)
    use_finbert = _should_use_finbert(model_mode, enable_finbert)

    try:
        cache = _read_cache()
    except Exception:
        cache = {}

    key = _cache_key(symbol_key, query, model_mode, use_finbert)
    cache_entry = cache.get(key)
    if cache_entry and (time.time() - cache_entry.get("ts", 0) < CACHE_TTL):
        return cache_entry

    headlines = _get_headlines_for_query(query, use_newsapi=use_newsapi, max_items=max_headlines)
    vader_sentiment = score_texts(headlines) if headlines else 0.0
    dl_result = {"available": False, "reason": "FinBERT not requested."}
    if model_mode in {"dl", "hybrid"} and use_finbert:
        dl_result = score_news_with_finbert(headlines)

    sentiment = dl_result.get("score", vader_sentiment) if dl_result.get("available") else vader_sentiment
    events = detect_stock_events(headlines)
    rule_score = compute_news_score(sentiment, events)
    rule_action = action_from_score(rule_score, len(headlines))
    rule_confidence = confidence_from_score(rule_score, len(headlines))

    ml_result = {"available": False, "reason": "ML classifier not requested."}
    if model_mode in {"ml", "hybrid"}:
        ml_result = predict_news_action_ml(headlines)

    score = rule_score
    action = rule_action
    confidence = rule_confidence

    if model_mode == "ml" and ml_result.get("available"):
        action = ml_result["action"]
        confidence = int(ml_result.get("confidence", rule_confidence))
        score = round(_action_to_numeric(action, confidence), 4)
    elif model_mode == "hybrid" and ml_result.get("available"):
        ml_score = _action_to_numeric(ml_result["action"], int(ml_result.get("confidence", 50)))
        score = round((0.45 * rule_score) + (0.55 * ml_score), 4)
        action = action_from_score(score, len(headlines))
        confidence = int(round((rule_confidence + int(ml_result.get("confidence", 50))) / 2))

    result = {
        "symbol": symbol_key,
        "name": config.get("name", symbol_key),
        "exchange_symbol": config.get("symbol", symbol_key),
        "asset_type": config.get("type", "stock"),
        "action": action,
        "prediction": {"TO_BUY": 1, "HOLD": 0, "SELL": -1}[action],
        "score": score,
        "confidence": confidence,
        "reasons": _reason_summary(action, score, sentiment, events),
        "model_mode": model_mode,
        "models": {
            "rules": {
                "available": True,
                "action": rule_action,
                "score": rule_score,
                "confidence": rule_confidence,
                "sentiment_backend": "finbert" if dl_result.get("available") else "vader",
                "vader_sentiment": round(vader_sentiment, 4),
                "effective_sentiment": round(sentiment, 4),
            },
            "ml": ml_result,
            "dl": dl_result,
        },
        "events": events,
        "headlines": headlines[:10],
        "headlines_count": len(headlines),
        "query": query,
        "ts": int(time.time()),
    }
    if ml_result.get("available"):
        result["reasons"].append(
            f"ML classifier {ml_result['action']} at {ml_result.get('confidence', 0)}%"
        )
    elif model_mode in {"ml", "hybrid"}:
        result["reasons"].append(str(ml_result.get("reason", "ML classifier unavailable.")))

    if dl_result.get("available"):
        result["reasons"].append(f"FinBERT sentiment {dl_result.get('score', 0):+.3f}")
    elif model_mode == "dl" or use_finbert:
        result["reasons"].append(str(dl_result.get("reason", "FinBERT unavailable.")))

    cache[key] = result
    try:
        _write_cache(cache)
    except Exception:
        logger.debug("Could not write news cache", exc_info=True)
    return result


def analyze_watchlist(
    assets: Optional[Dict[str, Dict[str, Any]]] = None,
    use_newsapi: bool = True,
    max_headlines: int = 25,
    model_mode: str = "hybrid",
    enable_finbert: Optional[bool] = None,
) -> Dict[str, Any]:
    """Analyze all watched stocks and group them for the dashboard columns."""
    assets = assets or MONITORED
    grouped = {action: [] for action in ACTIONS}
    errors = []

    for symbol_key in sorted(assets.keys()):
        try:
            signal = get_news_prediction(
                symbol_key,
                use_newsapi=use_newsapi,
                max_headlines=max_headlines,
                assets=assets,
                model_mode=model_mode,
                enable_finbert=enable_finbert,
            )
            grouped[signal["action"]].append(signal)
        except Exception as exc:
            errors.append({"symbol": symbol_key, "error": str(exc)})

    for signals in grouped.values():
        signals.sort(key=lambda item: (item.get("confidence", 0), abs(item.get("score", 0))), reverse=True)

    return {
        "generated_at": int(time.time()),
        "mode": model_mode,
        "finbert_enabled": _should_use_finbert(model_mode, enable_finbert),
        "groups": grouped,
        "errors": errors,
        "counts": {action: len(grouped[action]) for action in ACTIONS},
    }


if __name__ == "__main__":
    import sys

    keys = sys.argv[1:] or list(MONITORED.keys())
    for key in keys:
        print(get_news_prediction(key))
