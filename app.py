# app.py — Streamlit + Twelve Data + matplotlib (ilman CSV-latausta)
import os
import time
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------- Streamlit asetukset ----------
st.set_page_config(page_title="Osakekuvaaja (Twelve Data + matplotlib)", layout="centered")
st.title("📈 Osakekuvaaja — Twelve Data + matplotlib")

# ---------- HTTP-sessio ----------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/121.0.0.0 Safari/537.36"),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
})

# ---------- API-avain Streamlitin Secretsista ----------
API_KEY = st.secrets.get("TWELVEDATA_API_KEY")
if not API_KEY:
    st.error("⚠️ Twelve Data API Key puuttuu! Lisää se Streamlit 'Secrets' -asetuksiin.")
    st.stop()

# ---------- Aikajaksot ----------
PERIODS = {
    "1kk": dict(months=1),
    "3kk": dict(months=3),
    "6kk": dict(months=6),
    "YTD": "ytd",
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
        start = now - timedelta(days=days + 7)
    return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")

# ---------- Twelve Data haku ----------
def fetch_td_series(symbol: str, start_date: str, end_date: str, interval="1day") -> pd.Series:
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={quote_plus(symbol)}"
        f"&interval={interval}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&order=ASC&outputsize=5000"
        f"&apikey={quote_plus(API_KEY)}"
    )
    time.sleep(0.3)
    resp = SESSION.get(url, timeout=20)
    if not resp.ok:
        return pd.Series(dtype=float)

    js = resp.json()
    if "values" not in js:
        return pd.Series(dtype=float)

    df = pd.DataFrame(js["values"])
    if df.empty or "datetime" not in df.columns or "close" not in df.columns:
        return pd.Series(dtype=float)

    df["Date"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["Close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).set_index("Date").sort_index()
    return df["Close"]

# ---------- Symbolilista ----------
symbol_list = ["NOKIA.HE", "SAMPO.HE", "ELISA.HE", "AAPL", "MSFT", "SPY"]

left, right = st.columns([2,1])
symbol = left.selectbox("Valitse symboli", options=symbol_list, index=0)
custom = left.text_input("…tai kirjoita oma ticker (esim. NVDA, ADS.DE)")
period_label = right.selectbox("Aikajakso", list(PERIODS.keys()), index=4)

symbol = (custom.strip().upper() if custom.strip() else symbol).strip()

# ---------- Näytä kuvaaja ----------
if st.button("Näytä kuvaaja", type="primary"):
    start_date, end_date = compute_start_end(period_label)
    s = fetch_td_series(symbol, start_date, end_date, interval="1day")

    if s.empty:
        st.error(f"Twelve Data ei palauttanut dataa: {symbol}")
        st.stop()

    # --- Tilastot ---
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

    # --- Piirto matplotlibilla ---
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(s.index, s.values, label=f"{symbol}", linewidth=1.6)
    ax.axhline(mean, linestyle="--", color="green", label="Mean")
    for k, col in zip([1,2,3], ["blue","orange","red"]):
        ax.axhline(mean + k*std, linestyle="--", color=col)
        ax.axhline(mean - k*std, linestyle=":", color=col)
    ax.scatter(s.index[-1], last, zorder=5)

    if std == 0:
        title = f"{symbol} — {period_label}\nσ=0 (ei vaihtelua)"
    else:
        title = f"{symbol} — {period_label}\nP(|Z|≥|zₙ|): {(prob_tail*100):.3f}% | % ±1σ: {pct1:.1f}%"
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    st.pyplot(fig)

    # --- Info ---
    st.caption(
        f"Keskihinta: {mean:.2f} | ±1σ: [{mean-std:.2f}, {mean+std:.2f}] | "
        f"±2σ: [{mean-2*std:.2f}, {mean+2*std:.2f}] | "
        f"±3σ: [{mean-3*std:.2f}, {mean+3*std:.2f}]"
    )
