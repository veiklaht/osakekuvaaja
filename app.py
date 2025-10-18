# app.py — Streamlit + Twelve Data + matplotlib
# Ominaisuudet:
# - Lisää kuvaaja: kasaa ruudukkoon 3/row
# - Tyhjennä kaikki
# - Twelve Data API key: st.secrets["TWELVEDATA_API_KEY"]
# - Automaattinen symboli-fallback: .HE -> :XHEL -> ADR (esim. ELMUF)

import time
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------- Streamlit & HTTP ----------------
st.set_page_config(page_title="Osakekuvaaja (Twelve Data)", layout="wide")
st.title("📈 USA Osakekuvaaja")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (StockApp; +https://streamlit.app)",
    "Accept": "application/json,text/plain,*/*",
    "Connection": "keep-alive",
})

API_KEY = st.secrets.get("TWELVEDATA_API_KEY")
if not API_KEY:
    st.error("⚠️ Twelve Data API Key puuttuu! Lisää se Streamlit ‘Secrets’ -asetuksiin.")
    st.stop()

# ---------------- Aikajaksot ----------------
PERIODS = {
    "1kk": dict(months=1),
    "3kk": dict(months=3),
    "6kk": dict(months=6),
    "1v": dict(years=1),
    "5v": dict(years=5),
    "10v": dict(years=10),
    "15v": dict(years=15),
}

def compute_start_end(label: str):
    now = datetime.utcnow()
    if label == "YTD":
        start = datetime(now.year, 1, 1)
    else:
        spec = PERIODS[label]
        days = (spec.get("months", 0) * 30) + (spec.get("years", 0) * 365)
        start = now - timedelta(days=days + 7)  # pieni puskuri
    return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")

# ---------------- Symboli-fallbackit ----------------
TD_EXCHANGE_MAP = {
    "HE": "XHEL", "ST": "XSTO", "CO": "XCSE", "OL": "XOSL",
    "DE": "XETR", "F": "XFRA", "PA": "XPAR", "AS": "XAMS",
    "MI": "XMIL", "L": "XLON", "SW": "XSWX", "TO": "XTSE",
    "V": "XTSX"
}
ADR_FALLBACKS = {
    "NOKIA": ["NOK"], "ELISA": ["ELMUF","ELMAY"], "SAMPO": ["SAXPY"],
    "KNEBV": ["KNYJF"], "NESTE": ["NTOIF","NTOIY"], "UPM": ["UPMKY"],
    "FORTUM": ["FOJCF"], "METSB": ["MTSAF"], "KESKOB": ["KKOYF","KKOYB"]
}

def td_symbol_candidates(yahoo_like: str) -> list[str]:
    s = yahoo_like.strip().upper()
    cands = [s]
    base = s.split(".")[0]
    if "." in s:
        suf = s.rsplit(".", 1)[-1]
        ex = TD_EXCHANGE_MAP.get(suf)
        if ex:
            cands.append(f"{base}:{ex}")
    cands += ADR_FALLBACKS.get(base, [])
    # poista duplikaatit säilyttäen järjestyksen
    seen, out = set(), []
    for t in cands:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ---------------- Twelve Data haku ----------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_td_series(symbol: str, start_date: str, end_date: str, interval="1day") -> tuple[pd.Series, str]:
    """Palauttaa (sarja, käytetty_symboli). Kokeilee useita kandidaatteja järjestyksessä."""
    for sym in td_symbol_candidates(symbol):
        url = (
            "https://api.twelvedata.com/time_series"
            f"?symbol={quote_plus(sym)}"
            f"&interval={interval}"
            f"&start_date={start_date}"
            f"&end_date={end_date}"
            f"&order=ASC&outputsize=5000"
            f"&apikey={quote_plus(API_KEY)}"
        )
        time.sleep(0.25)  # pieni throttle
        resp = SESSION.get(url, timeout=20)
        if not resp.ok:
            continue
        js = resp.json()
        if "values" not in js:
            continue
        df = pd.DataFrame(js["values"])
        if df.empty or "datetime" not in df.columns or "close" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["Close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).set_index("Date").sort_index()
        s = df["Close"].astype(float)
        if not s.empty:
            return s, sym
    return pd.Series(dtype=float), symbol

# ---------------- Piirto ----------------
def plot_one_chart(s: pd.Series, symbol_shown: str, period_label: str):
    mean = float(s.mean())
    std  = float(s.std(ddof=1)) if s.std(ddof=1) != 0 else 0.0
    last = float(s.iloc[-1])

    if std == 0:
        prob_tail = float("nan")
        pct1 = float("nan")
    else:
        z = (last - mean) / std
        prob_tail = 2 * (1 - norm.cdf(abs(z)))
        pct1 = (s.between(mean - std, mean + std)).mean() * 100

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(s.index, s.values, label=symbol_shown, linewidth=1.6)
    ax.axhline(mean, linestyle="--", color="green", label="Mean")
    for k, col in zip([1, 2, 3], ["blue", "orange", "red"]):
        ax.axhline(mean + k*std, linestyle="--", color=col)
        ax.axhline(mean - k*std, linestyle=":", color=col)
    ax.scatter(s.index[-1], last, zorder=5)

    title = (f"{symbol_shown} — {period_label}\nP(|Z|≥|zₙ|): {prob_tail*100:.3f}% | % ±1σ: {pct1:.1f}%"
             if std != 0 else f"{symbol_shown} — {period_label}\nσ=0 (ei vaihtelua)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

    st.caption(
        f"Keskihinta: {mean:.2f} | ±1σ: [{mean-std:.2f}, {mean+std:.2f}] | "
        f"±2σ: [{mean-2*std:.2f}, {mean+2*std:.2f}] | "
        f"±3σ: [{mean-3*std:.2f}, {mean+3*std:.2f}]"
    )

# ---------------- UI: valinnat + napit ----------------
default_symbols = ["NOKIA.HE", "SAMPO.HE", "ELISA.HE", "AAPL", "MSFT", "SPY"]

top_left, top_mid, top_right = st.columns([2, 1.2, 1])
symbol = top_left.selectbox("Symboli", options=default_symbols, index=0)
custom = top_left.text_input("…tai oma ticker (esim. NVDA, ADS.DE)")
period_label = top_mid.selectbox("Aikajakso", list(PERIODS.keys()), index=4)
add_btn = top_right.button("➕ Lisää kuvaaja", type="primary")
clear_btn = top_right.button("🗑️ Tyhjennä kaikki")

# Muuta valinta, jos oma ticker annettu
final_symbol = (custom.strip().upper() if custom.strip() else symbol).strip()

# ---------------- Session state lista kuvista ----------------
if "plots" not in st.session_state:
    st.session_state.plots = []  # lista dict: {"symbol":..., "period":...}

if add_btn:
    st.session_state.plots.append({"symbol": final_symbol, "period": period_label})

if clear_btn:
    st.session_state.plots.clear()

# ---------------- Ruudukko: 3 kuvaajaa per rivi ----------------
plots = st.session_state.plots
if not plots:
    st.info("Vinkki: valitse symboli ja aikajakso ylhäältä ja paina **Lisää kuvaaja**. Ruudukossa on 3 kuvaajaa per rivi.")
else:
    # Piirrä riveissä (3 per rivi)
    for i in range(0, len(plots), 3):
        row = plots[i:i+3]
        cols = st.columns(3)
        for col, item in zip(cols, row):
            with col:
                start_date, end_date = compute_start_end(item["period"])
                with st.spinner(f"Haetaan dataa: {item['symbol']} ({item['period']})"):
                    s, used_sym = fetch_td_series(item["symbol"], start_date, end_date, interval="1day")
                if s.empty:
                    st.error(f"Ei dataa: {item['symbol']}")
                else:
                    if used_sym != item["symbol"]:
                        st.caption(f"Haettu symbolilla: **{used_sym}**")
                    plot_one_chart(s, used_sym, item["period"])
