# app.py — Osakekuvaaja (Streamlit)
# Lähdejärjestys: 1) Yahoo CSV (query1/query2 + crumb)  2) Stooq CSV  3) Twelve Data API
# Ei yfinancea (ei YFRateLimitErroria)

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
import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Osakekuvaaja", layout="centered")
st.set_option("client.showErrorDetails", True)

# ---------------- Throttle & Backoff ----------------
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

# ---------------- HTTP-sessio ----------------
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

# ---------------- CSV-lista ----------------
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

# ---------------- UI ----------------
years_options = {"1v": 1, "5v": 5, "10v": 10, "15v": 15}

st.title("📈 Yksinkertainen osakekuvaaja")
st.write("Data: Yahoo CSV → Stooq → Twelve Data (API). Jos Yahoo/Stooq blokkaa, Twelve Data varmistaa tuloksen.")

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

# ---------------- Ticker-variantit ----------------
def symbol_variants(sym: str) -> list[str]:
    s = sym.strip().upper()
    out = [s]
    base = s.split(".")[0]
    out.append(base)  # joskus base ilman päätettä toimii

    HE_ADR = {
        "NOKIA":  ["NOK"],            # Nokia ADR (NYSE)
        "ELISA":  ["ELMUF", "ELMAY"],
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

# ---------------- Yahoo CSV helperit ----------------
@st.cache_data(ttl=3600)
def get_crumb_and_cookie(host: str = "query1") -> tuple[str | None, requests.cookies.RequestsCookieJar]:
    url = f"https://{host}.finance.yahoo.com/v1/test/getcrumb"
    resp = SESSION.get(url, timeout=10)
    if not resp.ok or not resp.text.strip():
        other = "query2" if host == "query1" else "query1"
        resp = SESSION.get(f"https://{other}.finance.yahoo.com/v1/test/getcrumb", timeout=10)
    crumb = resp.text.strip() if resp.ok else None
    return (crumb if crumb else None), SESSION.cookies

def build_csv_url(ticker: str, p1: int, p2: int, interval: str, host: str, crumb: str | None) -> str:
    base = f"https://{host}.finance.yahoo.com/v7/finance/download/{quote_plus(ticker)}"
    qs = f"?period1={p1}&period2={p2}&interval={interval}&events=history&includeAdjustedClose=true"
    if crumb:
        qs += f"&crumb={quote_plus(crumb)}"
    return base + qs

def try_fetch_yahoo_csv(ticker: str, start_dt: datetime, end_dt: datetime, interval: str, attempts: int = 2):
    p1, p2 = int(start_dt.timestamp()), int(end_dt.timestamp())
    hosts = ["query1", "query2"]
    tried_urls = []

    for host in hosts:
        crumb, _cookies = get_crumb_and_cookie(host)
        for attempt in range(1, attempts + 1):
            throttle(2.0)
            url = build_csv_url(ticker, p1, p2, interval, host, crumb)
            tried_urls.append(f"{host}:{ticker} [{interval}]")
            resp = SESSION.get(url, timeout=15)

            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                retry_after = float(ra) if ra and ra.isdigit() else None
                backoff_sleep(attempt=attempt, retry_after=retry_after)
                continue

            if resp.ok and resp.text and "Date,Open,High,Low,Close" in resp.text:
                df = pd.read_csv(io.StringIO(resp.text))
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                    return df, tried_urls

            backoff_sleep(attempt=attempt)
    return None, tried_urls

def series_from_ohlc_df(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df["Adj Close"] if "Adj Close" in df.columns else df.get("Close")
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    return s

# ---------------- Stooq CSV helperit ----------------
def to_stooq_symbols(ticker: str) -> list[str]:
    t = ticker.upper()
    out = [f"{t.lower()}.us"]  # yleinen US-listaus/ADR
    base = t.split(".")[0]
    ADR_MAP = {
        "NOKIA.HE": ["nok.us"],
        "NOKIA":    ["nok.us"],
        "NOK":      ["nok.us"],
        "AAPL":     ["aapl.us"],
        "MSFT":     ["msft.us"],
        "AMZN":     ["amzn.us"],
        "META":     ["meta.us"],
        "NVDA":     ["nvda.us"],
        "TSLA":     ["tsla.us"],
        "SPY":      ["spy.us"],
        "QQQ":      ["qqq.us"],
        "VTI":      ["vti.us"],
        "ADDYY":    ["addyy.us"],
    }
    if t in ADR_MAP:
        out = ADR_MAP[t] + out
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def try_fetch_stooq_csv(stooq_symbol: str, interval: str, attempts: int = 2):
    i_map = {"1d": "d", "1wk": "w"}
    ii = i_map.get(interval, "d")
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i={ii}"
    for attempt in range(1, attempts + 1):
        throttle(1.0)
        resp = SESSION.get(url, timeout=15)
        if resp.ok and resp.text and "Date,Open,High,Low,Close,Volume" in resp.text:
            df = pd.read_csv(io.StringIO(resp.text))
            if "Date" in df.columns and not df.empty:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                return df
        backoff_sleep(attempt=attempt, base=0.6, cap=4.0)
    return None

# ---------------- Twelve Data helperit ----------------
def to_twelvedata_symbols(ticker: str) -> list[str]:
    """
    Twelve Data tukee suoraan 'NOKIA.HE', 'ELISA.HE', jne.
    Lisäksi kokeile yleisiä ADR/US-varianteja.
    """
    t = ticker.upper().strip()
    out = [t]
    base = t.split(".")[0]
    TD_MAP = {
        "NOKIA.HE": ["NOKIA.HE","NOK"],  # suora HE + ADR
        "NOKIA":    ["NOKIA.HE","NOK"],
        "NOK":      ["NOK"],
        "ELISA.HE": ["ELISA.HE","ELMUF","ELMAY"],
        "SAMPO.HE": ["SAMPO.HE","SAXPY"],
        "AAPL":     ["AAPL"],
        "MSFT":     ["MSFT"],
        "AMZN":     ["AMZN"],
        "SPY":      ["SPY"],
        "ADS.DE":   ["ADS.DE","ADDYY"],
    }
    if t in TD_MAP:
        out = TD_MAP[t] + out
    # poista duplikaatit
    seen, uniq = set(), []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def try_fetch_twelvedata(sym: str, years: int, interval: str, apikey: str):
    """
    Twelve Data: https://api.twelvedata.com/time_series
    interval: 1day / 1week
    outputsize: riittävän iso, jotta kattaa vuosia
    """
    int_map = {"1d": "1day", "1wk": "1week"}
    itv = int_map.get(interval, "1day")
    outputsize = 5000  # maksimoi historian
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={quote_plus(sym)}"
        f"&interval={itv}"
        f"&outputsize={outputsize}"
        "&order=ASC"
        f"&apikey={quote_plus(apikey)}"
    )
    throttle(1.0)
    resp = SESSION.get(url, timeout=20)
    if not resp.ok:
        return None
    js = resp.json()
    if "values" not in js:
        return None
    df = pd.DataFrame(js["values"])
    if df.empty or "datetime" not in df.columns or "close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["Close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["Date","Close"]).set_index("Date").sort_index()
    # rajaa haluttuun vuosimäärään
    cutoff = df.index.max() - pd.DateOffset(years=years)
    df = df.loc[df.index >= cutoff]
    # kopioi Close myös Adj Close -kolumniksi yhteensopivuuden vuoksi
    df["Adj Close"] = df["Close"]
    return df

# ---------------- Pää-haku: Yahoo CSV -> Stooq -> Twelve Data ----------------
@st.cache_data(ttl=600)
def fetch_series(sym: str, years: int):
    debug_lines = []
    candidates = symbol_variants(sym)
    intervals = ["1d", "1wk"]
    periods = [years, years + 1]

    # 1) Yahoo CSV
    for cand in candidates:
        for per in periods:
            for itv in intervals:
                end_dt = datetime.utcnow() + timedelta(days=2)
                start_dt = end_dt - timedelta(days=int(per * 365 + 10))
                debug_lines.append(f"{cand} [{per}y {itv}] (Yahoo CSV)")
                y_df, urls = try_fetch_yahoo_csv(cand, start_dt, end_dt, itv, attempts=2)
                debug_lines += [f"  -> {u}" for u in urls]
                if y_df is not None and not y_df.empty:
                    s = series_from_ohlc_df(y_df)
                    if not s.empty:
                        return s, cand, debug_lines

    # 2) Stooq CSV
    for cand in candidates:
        stooq_syms = to_stooq_symbols(cand)
        for stq in stooq_syms:
            for itv in intervals:
                debug_lines.append(f"{cand} [{itv}] (Stooq: {stq})")
                s_df = try_fetch_stooq_csv(stq, itv, attempts=2)
                if s_df is not None and not s_df.empty:
                    s = series_from_ohlc_df(s_df)
                    if not s.empty:
                        return s, f"{cand} (via {stq})", debug_lines

    # 3) Twelve Data (API KEY vaaditaan)
    td_key = st.secrets.get("TWELVEDATA_API_KEY")
    if td_key:
        for cand in candidates:
            td_syms = to_twelvedata_symbols(cand)
            for sym2 in td_syms:
                for itv in intervals:
                    debug_lines.append(f"{cand} [{itv}] (Twelve Data: {sym2})")
                    td_df = try_fetch_twelvedata(sym2, years, itv, td_key)
                    if td_df is not None and not td_df.empty:
                        s = series_from_ohlc_df(td_df)
                        if not s.empty:
                            return s, f"{cand} (via TwelveData:{sym2})", debug_lines
    else:
        debug_lines.append("Twelve Data API key puuttuu: lisää TWELVEDATA_API_KEY secretsiin.")

    return pd.Series(dtype=float), None, debug_lines

# ---------------- Laskenta & Piirto ----------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

if st.button("Näytä kuvaaja", type="primary"):
    with st.spinner("Haetaan dataa…"):
        s, used, tried = fetch_series(symbol, years)

    if show_debug:
        st.code("\n".join(tried), language="text")

    if s.empty:
        st.warning(
            "Yksikään lähde ei palauttanut dataa: **{}**.\n\n"
            "Yritetyt yhdistelmät:\n- {}\n\n"
            "Vinkit: lisää Twelve Data API -avain (Settings → Secrets), tai kokeile US/ADR-tickeriä."
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
