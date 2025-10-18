# app.py — Osakekuvaaja (Streamlit), robusti Yahoo-haku + CSV-lista

import math
import io
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

# ------------------------------
# Yleisasetukset
# ------------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Osakekuvaaja", layout="centered")

# ------------------------------
# A) Yhteinen requests.Session + "selain" headerit
# ------------------------------
SESSION = requests.Session()
SESSION.headers.update({
    # Moderni desktop-selaimen UA vähentää Yahoo HTML/403-vastauksia
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

# ------------------------------
# CSV-luku: ticket_base.csv tai tickers_base.csv
# ------------------------------
root = Path(__file__).parent
df = None
for name in ["ticket_base.csv", "tickers_base.csv"]:
    p = root / name
    if p.exists():
        df = pd.read_csv(p)
        break

needed = {"symbol", "name", "exchange", "country", "asset_class", "currency", "notes"}
if df is None:
    # Fallback-mini CSV jos oma tiedosto puuttuu
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

# ------------------------------
# UI: valinnat
# ------------------------------
years_options = {"1v": 1, "5v": 5, "10v": 10, "15v": 15}

st.title("📈 Yksinkertainen osakekuvaaja")
st.write(
    "Valitse listasta tai kirjoita oma ticker. Data haetaan Yahoo Financesta. "
    "Näytetään keskihinta sekä ±1/2/3σ-rajat ja kahden hännän todennäköisyys "
    "(kuinka epätodennäköinen nykyinen poikkeama on normaalijakaumassa)."
)

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

# ------------------------------
# B) Ticker-variantit (HE→ADR, DE→F, ym.)
# ------------------------------
def symbol_variants(sym: str) -> list[str]:
    s = sym.strip().upper()
    out = [s]
    base = s.split(".")[0]

    # Yleisiksi vaihtoehdoiksi mukaan myös "base" ilman päätettä (joissain listauksissa toimii)
    out.append(base)

    # Helsingin pörssi → mahdolliset ADR/OTC -vastineet
    HE_ADR = {
        "NOKIA":  ["NOK"],            # Nokia ADR (NYSE)
        "ELISA":  ["ELMUF", "ELMAY"], # Elisa OTC
        "SAMPO":  ["SAXPY"],          # Sampo ADR
        "KNEBV":  ["KNYJF"],          # Kone B OTC
        "NESTE":  ["NTOIF", "NTOIY"],
        "FORTUM": ["FOJCF"],
        "UPM":    ["UPMKY"],
        "METSB":  ["MTSAF"],          # Metsä Board B OTC
        "KESKOB": ["KKOYF", "KKOYB"],
        "WRT1V":  ["WRTBY"],          # Wärtsilä ADR
    }
    if s.endswith(".HE") and base in HE_ADR:
        out += HE_ADR[base]

    # Saksa (XETRA) → Frankfurt -vaihtoehto
    if s.endswith(".DE"):
        out.append(f"{base}.F")
        if base == "ADS":  # Adidas ADR
            out.append("ADDYY")

    # Duplikaatit pois
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

# ------------------------------
# C) Robustisti & välimuistilla: download → history → CSV-endpoint
# ------------------------------
@st.cache_data(ttl=600)
def fetch_series(sym: str, years: int) -> tuple[pd.Series, str | None, list[str]]:
    """
    Palauttaa (pd.Series, käytetty_symboli, yritetyt_vaihtoehdot)
    Yrittää yfinance.download → Ticker.history → suora Yahoo CSV
    ja testaa myös symbolivariaatiot, 1d/1wk sekä years/years+1.
    """
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

    tried: list[str] = []
    candidates = symbol_variants(sym)
    intervals = ["1d", "1wk"]
    periods = [years, years + 1]

    for cand in candidates:
        for per in periods:
            for itv in intervals:
                tried.append(f"{cand} [{per}y {itv}]")

                # 1) yfinance.download
                try:
                    df = yf.download(
                        cand, period=f"{per}y", interval=itv,
                        auto_adjust=False, progress=False, threads=False,
                        session=SESSION,
                    )
                    s = _series_from_df(df)
                    if not s.empty:
                        return s, cand, tried
                except Exception:
                    time.sleep(0.8)

                # 2) Ticker.history
                try:
                    t = yf.Ticker(cand, session=SESSION)
                    hist = t.history(period=f"{per}y", interval=itv, auto_adjust=False)
                    s = _series_from_df(hist)
                    if not s.empty:
                        return s, cand, tried
                except Exception:
                    time.sleep(0.8)

                # 3) Suora Yahoo CSV -endpoint
                try:
                    # laske epoch-ajat: per vuotta taakse + puskuria
                    end_dt = datetime.utcnow() + timedelta(days=2)
                    start_dt = end_dt - timedelta(days=int(per * 365 + 10))
                    period1 = int(start_dt.timestamp())
                    period2 = int(end_dt.timestamp())

                    url = (
                        "https://query1.finance.yahoo.com/v7/finance/download/"
                        f"{quote_plus(cand)}"
                        f"?period1={period1}&period2={period2}"
                        f"&interval={itv}&events=history&includeAdjustedClose=true"
                    )
                    resp = SESSION.get(url, timeout=15)
                    if resp.ok and resp.text and "Date,Open,High,Low,Close" in resp.text:
                        csv_df = pd.read_csv(io.StringIO(resp.text))
                        if "Date" in csv_df.columns:
                            csv_df["Date"] = pd.to_datetime(csv_df["Date"], errors="coerce")
                            csv_df = csv_df.dropna(subset=["Date"]).set_index("Date").sort_index()
                            s = _series_from_df(csv_df)
                            if not s.empty:
                                return s, cand, tried
                except Exception:
                    time.sleep(0.8)

    return pd.Series(dtype=float), None, tried

# ------------------------------
# D) UI-toiminto: nappi → haku → laskenta → kuvaaja
# ------------------------------
def norm_cdf(x: float) -> float:
    # N(0,1) CDF ilman SciPyä
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

if st.button("Näytä kuvaaja", type="primary"):
    s, used, tried = fetch_series(symbol, years)
    if s.empty:
        st.warning(
            "Ei saatavilla dataa: **{}**.\n\n"
            "Yritetyt vaihtoehdot:\n- {}\n\n"
            "Vinkit: kokeile toista markkinapäätettä (esim. `.F` Saksaan) tai ADR:ää (esim. `NOK`)."
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
        prob_tail = 2 * (1 - norm_cdf(abs(z)))   # kahden hännän todennäköisyys
        pct1 = (s.between(mean - std, mean + std)).mean() * 100

    # Kevyt piirto Streamlitin omalla kaaviolla
    st.line_chart(s, height=360)

    # Info
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
