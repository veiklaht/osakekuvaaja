# app.py — Osakekuvaaja (Streamlit), CSV-first haku + rate-limit backoff + CSV-lista

import io
import math
import random
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Osakekuvaaja", layout="centered")
st.set_option("client.showErrorDetails", True)

# --- throttling, ettei ammuta liikaa pyyntöjä per käyttäjä ---
if "last_call_ts" not in st.session_state:
    st.session_state.last_call_ts = 0.0

def throttle(min_interval=2.0):
    now = time.time()
    delta = now - st.session_state.last_call_ts
    if delta < min_interval:
        time.sleep(min_interval - delta)
    st.session_state.last_call_ts = time.time()

def backoff_sleep(attempt: int, retry_after: float | None = None, base: float = 0.8, cap: float = 8.0):
    if retry_after is not None:
        time.sleep(min(cap, max(0.0, retry_after)))
        return
    sleep_s = min(cap, base * (2 ** (attempt - 1)))
    sleep_s *= (0.8 + 0.4 * random.random())  # jitter
    time.sleep(sleep_s)

# --- yhteinen sessio “selain”-headeilla (auttaa Yahooa vastamaan siivosti) ---
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
})

# =========================
# CSV-LISTA (ticket_base.csv / tickers_base.csv)
# =========================
root = Path(__file__).parent
df = None
for name in ["ticket_base.csv", "tickers_base.csv"]:
    p = root / name
    if p.exists():
        df = pd.read_csv(p)
        break

needed = {"symbol", "name", "exchange", "country", "asset_class", "currency", "notes"}
if df is None:
    df = pd.DataFrame({
        "symbol": ["NOKIA.HE", "SAMPO.HE", "ELISA.HE", "AAPL", "MSFT", "SPY"],
        "name":   ["Nokia",    "Sampo",    "Elisa",    "Apple", "Microsoft", "SPDR S&P 500 ETF"],
        "exchange": ["Nasdaq Helsinki","Nasdaq Helsinki","Nasdaq Helsinki","NASDAQ","NASDAQ","NYSE Arca"],
        "country":  ["FI","FI","FI","US","US","US"],
        "asset_class": ["Equity","Equity","Equity","Equity","Equity","ETF"],
        "currency": ["EUR","EUR","EUR","USD","USD","USD"],
        "notes": ["","","","","",""]
    })
else:
    missing = needed - set(df.columns)
    if missing:
        st.error(f"CSV missing columns: {missing}. Keep header as: {sorted(needed)}")
        st.stop()

df = df.fillna("")
df["label"] = df.apply(
    lambda r: f'{r["symbol"]} — {r["name"]} '
              f'({", ".join([x for x in [r["exchange"], r["country"], r["currency"], r["asset_class"]] if x])})',
    axis=1
)

# =========================
# UI
# =========================
years_options = {"1v": 1, "5v": 5, "10v": 10, "15v": 15}

st.title("📈 Yksinkertainen osakekuvaaja")
st.write("Valitse listasta tai kirjoita oma ticker. Data haetaan Yahoo Financesta (CSV-reitti ensin).")

col_dbg, _ = st.columns([1,3])
show_debug = col_dbg.checkbox("Näytä debug-tiedot", value=False)

col0, _ = st.columns([1, 3])
asset = col0.selectbox("Asset class", options=["Kaikki"] + sorted(df["asset_class"].unique().tolist()), index=0)

dfv = df if asset == "Kaikki" else df[df["asset_class"] == asset]
dfv = dfv.sort_values("label")

col1, col2 = st.columns([2, 1])
symbol = col1.selectbox(
    "Symboli (kirjoita hakeaksesi)",
    options=dfv["symbol"].tolist(),
    index=0 if not dfv.empty else None,
    format_func=lambda s: dfv.loc[dfv["symbol"] == s, "label"].iloc[0] if s in dfv["symbol"].values else s,
    placeholder="Esim. NOKIA.HE, SXR8.DE, ^GSPC"
)
custom = col1.text_input("…tai anna oma ticker (Enter)")
years_label = col2.selectbox("Aikajakso", list(years_options.keys()), index=0)

symbol = (custom.strip().upper() if custom.strip() else symbol).strip()
years = years_options[years_label]

# =========================
# TICKER-VARIANTIT
# =========================
def symbol_variants(sym: str) -> list[str]:
    s = sym.strip().upper()
    out = [s]
    base = s.split(".")[0]
    out.append(base)  # joskus base ilman päätettä toimii

    HE_ADR = {
        "NOKIA":  ["NOK"],            # Nokia ADR (NYSE)
        "ELISA":  ["ELMUF", "ELMAY"], # Elisa OTC
        "SAMPO":  ["SAXPY"],
        "KNEBV":  ["KNYJF"],
        "NESTE":  ["NTOIF", "NTOIY"],
        "FORTUM": ["FOJCF"],
        "UPM":    ["UPMKY"],
        "METSB":  ["MTSAF"],
        "KESKOB": ["KKOYF", "KKOYB"],
        "WRT1V":  ["WRTBY"],
    }
    if s.endswith(".HE") and base in HE_ADR:
        out += HE_ADR[base]

    if s.endswith(".DE"):
        out.append(f"{base}.F")
        if base == "ADS":
            out.append("ADDYY")

    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

# =========================
# DATAHAKU: CSV -> (fallback) yfinance
# =========================
def _series_from_df(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df["Adj Close"] if "Adj Close" in df.columns else df.get("Close")
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    s.index = pd.to_datetime(s.index)
    return s

@st.cache_data(ttl=600)
def fetch_series(sym: str, years: int) -> tuple[pd.Series, str | None, list[str]]:
    """
    CSV-first: yritetään Yahoo CSV -endpointia ensin (usein vähemmän rajoituksia),
    sitten vasta yfinance.download / Ticker.history. Testaa myös 1d/1wk ja years/years+1.
    Palauttaa (sarja, käytetty_symboli, debug-yritykset).
    """
    tried: list[str] = []
    candidates = symbol_variants(sym)
    intervals = ["1d", "1wk"]
    periods = [years, years + 1]
    retries = 2

    # --- 1) CSV ensin ---
    for cand in candidates:
        for per in periods:
            for itv in intervals:
                tried.append(f"{cand} [{per}y {itv}] (CSV)")
                for attempt in range(1, retries + 1):
                    try:
                        throttle(2.0)
                        end_dt = datetime.utcnow() + timedelta(days=2)
                        start_dt = end_dt - timedelta(days=int(per * 365 + 10))
                        p1, p2 = int(start_dt.timestamp()), int(end_dt.timestamp())
                        url = (
                            "https://query1.finance.yahoo.com/v7/finance/download/"
                            f"{quote_plus(cand)}"
                            f"?period1={p1}&period2={p2}"
                            f"&interval={itv}&events=history&includeAdjustedClose=true"
                        )
                        resp = SESSION.get(url, timeout=15)

                        if resp.status_code == 429:
                            ra = resp.headers.get("Retry-After")
                            retry_after = float(ra) if ra and ra.isdigit() else None
                            backoff_sleep(attempt=attempt, retry_after=retry_after)
                            continue

                        if resp.ok and resp.text and "Date,Open,High,Low,Close" in resp.text:
                            csv_df = pd.read_csv(io.StringIO(resp.text))
                            if "Date" in csv_df.columns:
                                csv_df["Date"] = pd.to_datetime(csv_df["Date"], errors="coerce")
                                csv_df = csv_df.dropna(subset=["Date"]).set_index("Date").sort_index()
                                s = _series_from_df(csv_df)
                                if not s.empty:
                                    return s, cand, tried
                    except Exception:
                        backoff_sleep(attempt=attempt)

    # --- 2) yfinance (fallback) ---
    for cand in candidates:
        for per in periods:
            for itv in intervals:
                tried.append(f"{cand} [{per}y {itv}] (yf.download)")
                for attempt in range(1, retries + 1):
                    try:
                        throttle(2.0)
                        df = yf.download(
                            cand, period=f"{per}y", interval=itv,
                            auto_adjust=False, progress=False, threads=False,
                            session=SESSION,
                        )
                        s = _series_from_df(df)
                        if not s.empty:
                            return s, cand, tried
                    except Exception:
                        backoff_sleep(attempt=attempt)

                tried.append(f"{cand} [{per}y {itv}] (yf.history)")
                for attempt in range(1, retries + 1):
                    try:
                        throttle(2.0)
                        t = yf.Ticker(cand, session=SESSION)
                        hist = t.history(period=f"{per}y", interval=itv, auto_adjust=False)
                        s = _series_from_df(hist)
                        if not s.empty:
                            return s, cand, tried
                    except Exception:
                        backoff_sleep(attempt=attempt)

    return pd.Series(dtype=float), None, tried

# =========================
# LASKENTA & PIIRTO
# =========================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

if st.button("Näytä kuvaaja", type="primary"):
    with st.spinner("Haetaan dataa Yahoo CSV -rajapinnasta…"):
        s, used, tried = fetch_series(symbol, years)

    if show_debug:
        st.code("\n".join(tried), language="text")

    if s.empty:
        st.warning(
            "Yahoo Finance ei palauttanut dataa: **{}**.\n\n"
            "Yritetyt vaihtoehdot:\n- {}\n\n"
            "Vinkit: odota hetki (rate limit), kokeile toista markkinapäätettä (esim. `.F` Saksaan) "
            "tai ADR:ää (esim. **NOK** Nokialle)."
            .format(symbol, "\n- ".join(tried))
        )
        st.stop()

    mean = float(s.mean())
    std  = float(s.std(ddof=1)) if s.std(ddof=1) != 0 else 0.0
    last = float(s.iloc[-1])

    if std == 0:
        prob_tail = float("nan")
        pct1 = float("nan")
    else:
        z = (last - mean) / std
        prob_tail = 2 * (1 - norm_cdf(abs(z)))
        pct1 = (s.between(mean - std, mean + std)).mean() * 100

    st.line_chart(s, height=360)

    if used and used != symbol:
        st.caption(f"Haettiin data tickerillä: **{used}**")

    st.caption(
        f"Keskihinta: {mean:.2f} | ±1σ: [{mean - std:.2f}, {mean + std:.2f}] | "
        f"±2σ: [{mean - 2*std:.2f}, {mean + 2*std:.2f}] | "
        f"±3σ: [{mean - 3*std:.2f}, {mean + 3*std:.2f}]"
    )

    prob_txt = "—" if not np.isfinite(prob_tail) else f"{prob_tail*100:.3f}%"
    pct_txt  = "—" if not np.isfinite(pct1) else f"{pct1:.1f}%"
    st.info(f"Poikkeaman todennäköisyys (|Z|≥|z|): **{prob_txt}** | % ajasta ±1σ: **{pct_txt}**")
