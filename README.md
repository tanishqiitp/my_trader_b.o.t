# My Trader B.O.T (Custom Freqtrade Fork)

This is my customized fork of [Freqtrade](https://github.com/freqtrade/freqtrade) with additional features:

✅ **News Sentiment Integration**  
- Fetches real-time news for selected assets (via NewsAPI / Google News RSS)  
- Performs sentiment analysis using VADER  
- Injects sentiment score into strategy for better buy/sell signals  

✅ **Custom Scripts**  
- `scripts/fetch_news.py` → refreshes sentiment cache  
- Easy to schedule with Windows Task Scheduler  

✅ **Example Strategy**  
- Combines SMA crossover with news sentiment filter  
- Only enters trades when technicals + positive news align  

---

### Quick Start

```bash
git clone https://github.com/<your-username>/my_trader_b.o.t.git
cd my_trader_b.o.t
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install vaderSentiment feedparser requests

# Fetch sentiment
python scripts\fetch_news.py BTC ETH
