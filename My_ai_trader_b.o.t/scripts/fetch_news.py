# scripts/fetch_news.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, sys, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path (one level up from /scripts)
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from user_data.data.news_sentiment import get_sentiment
except ImportError as e:
    logger.error("Could not import news_sentiment: %s", e)
    sys.exit(1)

def main(argv):
    symbols = argv[1:] or ["BTC", "ETH"]
    for s in symbols:
        try:
            score = get_sentiment(s)
            print(f"{s}: {score:.3f}")
        except Exception as ex:
            logger.exception("Failed to get sentiment for %s: %s", s, ex)

if __name__ == "__main__":
    main(sys.argv)
