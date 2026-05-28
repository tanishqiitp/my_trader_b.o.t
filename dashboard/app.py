import time

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="NewsTrader", layout="wide")


def api_get(path, **params):
    params = {key: value for key, value in params.items() if value is not None}
    response = requests.get(f"{API_URL}{path}", params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def api_post(path, payload=None):
    response = requests.post(f"{API_URL}{path}", json=payload or {}, timeout=60)
    response.raise_for_status()
    return response.json()


def render_signal_card(signal):
    score = signal.get("score", 0)
    confidence = signal.get("confidence", 0)
    headlines = signal.get("headlines", [])
    reasons = signal.get("reasons", [])

    with st.container(border=True):
        st.subheader(signal.get("symbol", "UNKNOWN"))
        st.caption(signal.get("name", signal.get("exchange_symbol", "")))
        st.metric("News Score", f"{score:+.3f}", f"{confidence}% confidence")
        st.caption(f"Mode: {signal.get('model_mode', 'hybrid')}")
        if reasons:
            st.write(" | ".join(reasons[:3]))
        models = signal.get("models", {})
        ml = models.get("ml", {})
        dl = models.get("dl", {})
        model_bits = []
        if ml.get("available"):
            model_bits.append(f"ML: {ml.get('action')} ({ml.get('confidence')}%)")
        if dl.get("available"):
            model_bits.append(f"DL: FinBERT {dl.get('score'):+.3f}")
        if model_bits:
            st.caption(" | ".join(model_bits))
        if headlines:
            with st.expander("Latest headlines", expanded=False):
                for headline in headlines[:5]:
                    st.write(f"- {headline}")


st.title("NewsTrader")
st.caption("News-only watchlist analyzer for BUY / HOLD / SELL categorization.")

left, right = st.columns([2, 1])

with left:
    manual_stocks = st.text_area(
        "Manual stock list",
        value="TCS, RELIANCE, INFY, HDFCBANK, SBIN",
        height=110,
        help="Use comma/newline symbols. You can also write Name:SYMBOL, for example Tata Motors:TATAMOTORS.NS.",
    )
    manual_col, save_col = st.columns(2)
    analyze_manual = manual_col.button("Analyze Manual List", use_container_width=True)
    save_manual = save_col.button("Save As Watchlist", use_container_width=True)

with right:
    st.write("Broker import")
    broker = st.selectbox("Broker", ["zerodha", "groww"])
    st.caption("Set ZERODHA_HOLDINGS_FILE or GROWW_HOLDINGS_FILE to a CSV/JSON export path, then import.")
    import_broker = st.button("Import Broker Export", use_container_width=True)
    model_mode = st.selectbox(
        "Model mode",
        ["hybrid", "rules", "ml", "dl"],
        help="Hybrid uses rules plus your trained ML model when available. DL uses FinBERT if installed.",
    )
    enable_finbert = st.checkbox("Use FinBERT in hybrid", value=False)
    refresh_seconds = st.number_input("Auto refresh seconds", min_value=0, max_value=3600, value=0, step=30)

status_slot = st.empty()

if save_manual:
    try:
        result = api_post("/watchlist/manual", {"stocks": manual_stocks, "persist": True})
        status_slot.success(f"Saved {result['count']} stocks to the watchlist.")
    except Exception as exc:
        status_slot.error(f"Could not save watchlist: {exc}")

if import_broker:
    try:
        result = api_post(f"/watchlist/broker/{broker}")
        if result.get("status") == "imported":
            status_slot.success(f"Imported {result.get('count', 0)} stocks from {broker}.")
        else:
            status_slot.warning(result.get("message", "Broker import is not configured."))
    except Exception as exc:
        status_slot.error(f"Broker import failed: {exc}")

try:
    if analyze_manual:
        data = api_get(
            "/signals",
            manual=manual_stocks,
            model_mode=model_mode,
            enable_finbert=enable_finbert if model_mode == "hybrid" else None,
        )
    else:
        data = api_get(
            "/signals",
            model_mode=model_mode,
            enable_finbert=enable_finbert if model_mode == "hybrid" else None,
        )
except Exception as exc:
    st.error(f"Could not reach the API at {API_URL}. Start it with: uvicorn api.app:app --reload --port 8000")
    st.exception(exc)
    st.stop()

counts = data.get("counts", {})
generated_at = data.get("generated_at")
if generated_at:
    st.caption(f"Last analyzed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(generated_at))}")

try:
    with st.expander("ML/DL status", expanded=False):
        st.json(api_get("/model/status"))
except Exception:
    pass

buy_col, hold_col, sell_col = st.columns(3)

groups = data.get("groups", {})
columns = [
    ("TO BUY", "TO_BUY", buy_col),
    ("HOLD", "HOLD", hold_col),
    ("SELL", "SELL", sell_col),
]

for label, key, column in columns:
    with column:
        st.header(f"{label} ({counts.get(key, 0)})")
        signals = groups.get(key, [])
        if not signals:
            st.info("No stocks in this group.")
        for signal in signals:
            render_signal_card(signal)

if data.get("errors"):
    with st.expander("Analysis errors"):
        st.json(data["errors"])

if refresh_seconds:
    time.sleep(int(refresh_seconds))
    st.rerun()
