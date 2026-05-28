"""Optional ML/DL helpers for news classification.

The API keeps working without these dependencies. Install `scikit-learn` to
train/use the classical ML classifier, and install `transformers` + `torch` only
when you want the FinBERT deep-learning sentiment backend.
"""

import csv
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

ACTION_TO_BUY = "TO_BUY"
ACTION_HOLD = "HOLD"
ACTION_SELL = "SELL"
VALID_ACTIONS = {ACTION_TO_BUY, ACTION_HOLD, ACTION_SELL}

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "news_action_classifier.joblib")
DEFAULT_TRAINING_FILE = os.path.join(os.path.dirname(__file__), "news_training_examples.csv")


def normalize_action(value: str) -> str:
    normalized = value.strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "BUY": ACTION_TO_BUY,
        "TOBUY": ACTION_TO_BUY,
        "TO_BUY": ACTION_TO_BUY,
        "BULLISH": ACTION_TO_BUY,
        "1": ACTION_TO_BUY,
        "HOLD": ACTION_HOLD,
        "NEUTRAL": ACTION_HOLD,
        "0": ACTION_HOLD,
        "SELL": ACTION_SELL,
        "SELLOUT": ACTION_SELL,
        "SELL_OUT": ACTION_SELL,
        "BEARISH": ACTION_SELL,
        "-1": ACTION_SELL,
    }
    action = aliases.get(normalized, normalized)
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown label '{value}'. Use TO_BUY, HOLD, or SELL.")
    return action


def text_from_headlines(headlines: Iterable[str]) -> str:
    return "\n".join(str(item).strip() for item in headlines if str(item).strip())


def model_status() -> Dict[str, Any]:
    return {
        "ml_model_path": MODEL_PATH,
        "ml_model_exists": os.path.exists(MODEL_PATH),
        "training_file": DEFAULT_TRAINING_FILE,
        "training_file_exists": os.path.exists(DEFAULT_TRAINING_FILE),
        "dl_backend": "FinBERT optional",
        "finbert_enabled_by_env": os.environ.get("NEWS_ENABLE_FINBERT", "").lower()
        in {"1", "true", "yes", "on"},
    }


def read_training_rows(csv_path: Optional[str] = None) -> List[Dict[str, str]]:
    path = csv_path or DEFAULT_TRAINING_FILE
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("Training CSV must contain columns: text,label")
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                rows.append({"text": text, "label": normalize_action(label)})
    if len(rows) < 15:
        raise ValueError("Need at least 15 labeled rows to train a useful starter model.")
    return rows


def train_news_classifier(csv_path: Optional[str] = None, model_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies first: pip install -r requirements-ml.txt") from exc

    rows = read_training_rows(csv_path)
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=8000,
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    multi_class="auto",
                ),
            ),
        ]
    )

    metrics: Dict[str, Any] = {"rows": len(rows), "labels": sorted(set(labels))}
    if len(set(labels)) > 1 and len(rows) >= 30:
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels if min(labels.count(label) for label in set(labels)) >= 2 else None,
        )
        pipeline.fit(x_train, y_train)
        metrics["holdout_accuracy"] = round(float(pipeline.score(x_test, y_test)), 4)
    else:
        pipeline.fit(texts, labels)
        metrics["holdout_accuracy"] = None

    target_path = model_path or MODEL_PATH
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    joblib.dump(pipeline, target_path)
    _load_ml_model.cache_clear()
    metrics["model_path"] = target_path
    return metrics


@lru_cache(maxsize=1)
def _load_ml_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import joblib

        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def predict_news_action_ml(headlines: List[str]) -> Dict[str, Any]:
    text = text_from_headlines(headlines)
    if not text:
        return {"available": False, "reason": "No headlines to classify."}

    model = _load_ml_model()
    if model is None:
        return {
            "available": False,
            "reason": "No trained ML model found. Train one with scripts/train_news_model.py.",
        }

    label = str(model.predict([text])[0])
    probabilities: Dict[str, float] = {}
    confidence = 50
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probs = model.predict_proba([text])[0]
        probabilities = {str(cls): round(float(prob), 4) for cls, prob in zip(classes, probs)}
        confidence = int(round(max(probs) * 100))

    return {
        "available": True,
        "backend": "tfidf_logistic_regression",
        "action": normalize_action(label),
        "confidence": confidence,
        "probabilities": probabilities,
    }


@lru_cache(maxsize=1)
def _load_finbert_pipeline():
    try:
        from transformers import pipeline
    except ImportError as exc:
        return {"error": "Install DL dependencies first: pip install -r requirements-dl.txt", "exc": exc}

    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as exc:
        return {"error": f"Could not load FinBERT: {exc}", "exc": exc}


def score_news_with_finbert(headlines: List[str], max_items: int = 8) -> Dict[str, Any]:
    texts = [item for item in headlines[:max_items] if item]
    if not texts:
        return {"available": False, "reason": "No headlines to score."}

    backend = _load_finbert_pipeline()
    if isinstance(backend, dict) and backend.get("error"):
        return {"available": False, "reason": backend["error"]}

    try:
        outputs = backend(texts, truncation=True)
    except Exception as exc:
        return {"available": False, "reason": f"FinBERT inference failed: {exc}"}

    score = 0.0
    label_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for output in outputs:
        label = str(output.get("label", "")).lower()
        confidence = float(output.get("score", 0.0))
        if "positive" in label:
            score += confidence
            label_counts["positive"] += 1
        elif "negative" in label:
            score -= confidence
            label_counts["negative"] += 1
        else:
            label_counts["neutral"] += 1

    normalized = round(score / len(outputs), 4) if outputs else 0.0
    return {
        "available": True,
        "backend": "ProsusAI/finbert",
        "score": normalized,
        "confidence": int(round(min(0.95, abs(normalized) + 0.35) * 100)),
        "label_counts": label_counts,
    }
