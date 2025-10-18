import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from pathlib import Path
import time
import requests

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; StockApp/1.0; +https://streamlit.app)"
})

st.set_page_config(page_title="Osakekuvaaja", layout="centered")

# --- Lue CSV lista (tuetaan ticket_base.csv tai tickers_base.csv) ---
root = Path(__file__).parent
for name in ["ticket_base.csv", "tickers_base.csv"]:
    path = root / name
    if path.exists():
        df = pd.read_csv(path)
        break
else:
    df = pd.DataFrame({
        "symbol":["NOKIA.HE","SAMPO.HE","ELISA.HE","AAPL","MSFT","SPY"],
        "name":["Nokia","Sampo","Elisa","Apple","Microsoft","SPDR S&P 500 ETF"],
        "exchange":["Nasdaq Helsinki","Nasdaq Helsinki","Nasdaq Helsinki","NASDAQ","NASDAQ","NYSE Arca"],
        "country":["FI","FI","FI","US","US","US"],
        "asset_class":["Equity","Equity","Equity","Equity","Equity","ETF"],
        "currency":["EUR","EUR","EUR","USD","USD","USD"],
        "notes":["","","","","",""]
    })

needed = {"symbol","name","exchange","country","asset_class","currency","notes"}
missing = needed - set(df.columns)
if missing:
    st.error(f"CSV missing columns: {missing}. Please keep header as: {sorted(needed)}")
    st.stop()

df = df.fillna("")
df["label"] = df.apply(
    lambda r: f'{r["symbol"]} — {r["name"]} ({", ".join([x for x in [r["exchange"], r["country"], r["currency"], r["asset_class"]] if x])})',
    axis=1
)

years_options = {"1v":1, "5v":5, "10v":10, "15v":15}

def norm_cdf(x: float) -> float:
    # N(0,1) CDF ilman SciPyä
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

st.title("📈 Yksinkertainen osakekuvaaja")
st.write("Valitse listasta tai kirjoita oma ticker. Data haetaan Yahoo Financesta ja näytetään keskihinta sekä ±1/2/3σ-rajat ja tail-todennäköisyys.")

# --- Filtteri + valinnat ---
col0, _ = st.columns([1,3])
asset = col0.selectbox("Asset class", options=["Kaikki"] + sorted(df["asset_class"].unique().tolist()), index=0)
dfv = df if asset == "Kaikki" else df[df["asset_class"] == asset]
dfv = dfv.sort_values("label")

col1, col2 = st.columns([2,1])
symbol = col1.selectbox(
    "Symboli (kirjoita hakeaksesi)",
    options=dfv["symbol"].tolist(),
    index=0 if not dfv.empty else None,
    format_func=lambda s: dfv.loc[dfv["symbol"]==s,"label"].iloc[0] if s in dfv["symbol"].values else s,
    placeholder="Esim. NOKIA.HE, SXR8.DE, ^GSPC"
)
custom = col1.text_input("…tai anna oma ticker (Enter)")
years_label = col2.selectbox("Aikajakso", list(years_options.keys()), index=0)

symbol = (custom.strip().upper() if custom.strip() else symbol).strip()
years = years_options[years_label]

# --- Datahaku ja laskenta ---
#--def fetch_series(sym: str, years: int) -> pd.Series:
#    data = yf.download(sym, period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
#    if data is None or data.empty:
#        return pd.Series(dtype=float)
#   s = data["Adj Close"] if "Adj Close" in data.columns else data.get("Close")
#    if isinstance(s, pd.DataFrame): s = s.squeeze()
#    return s.dropna()

def fetch_series(sym: str, years: int, retries: int = 3, pause: float = 1.0) -> pd.Series:
    """
    Robustisti hae päivädata:
      1) yf.download(session=SESSION)
      2) fallback: yf.Ticker(sym, session=SESSION).history(...)
    Sis. retryt & virheiden nielemisen. Palauttaa pd.Series tai tyhjän sarjan.
    """
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                sym,
                period=f"{years}y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                session=SESSION,
            )
            if df is not None and not df.empty:
                s = df["Adj Close"] if "Adj Close" in df.columns else df.get("Close")
                if isinstance(s, pd.DataFrame):
                    s = s.squeeze()
                s = s.dropna()
                if not s.empty:
                    return s
        except Exception:
            # odota hetki ja yritä uudelleen
            time.sleep(pause * attempt)

        # Fallback: Ticker.history
        try:
            t = yf.Ticker(sym, session=SESSION)
            hist = t.history(period=f"{years}y", interval="1d", auto_adjust=False)
            if hist is not None and not hist.empty:
                s = hist["Adj Close"] if "Adj Close" in hist.columns else hist.get("Close")
                if isinstance(s, pd.DataFrame):
                    s = s.squeeze()
                s = s.dropna()
                if not s.empty:
                    return s
        except Exception:
            time.sleep(pause * attempt)

    # Kaikki yritykset epäonnistuivat
    return pd.Series(dtype=float)
    


if st.button("Näytä kuvaaja", type="primary"):
    s = fetch_series(symbol, years)
    if s.empty:
        st.warning(f"Ei saatavilla dataa: {symbol}")
        st.stop()

    mean = float(s.mean())
    std  = float(s.std(ddof=1)) if s.std(ddof=1)!=0 else 0.0
    last = float(s.iloc[-1])

    # Tilastot
    if std == 0:
        prob_tail = float("nan")
        pct1 = float("nan")
    else:
        z = (last - mean) / std
        prob_tail = 2 * (1 - norm_cdf(abs(z)))
        pct1 = (s.between(mean-std, mean+std)).mean()*100

    # Piirto Streamlitin omalla kaaviolla (yksi käyrä)
    st.line_chart(s, height=340)

    # Sigma-viivat + mean erikseen tekstinä (kevyesti)
    st.caption(f"Keskihinta: {mean:.2f} | ±1σ: [{mean-std:.2f}, {mean+std:.2f}] | ±2σ: [{mean-2*std:.2f}, {mean+2*std:.2f}] | ±3σ: [{mean-3*std:.2f}, {mean+3*std:.2f}]")

    prob_txt = "—" if not np.isfinite(prob_tail) else f"{prob_tail*100:.3f}%"
    pct_txt  = "—" if not np.isfinite(pct1) else f"{pct1:.1f}%"
    st.info(f"Poikkeaman todennäköisyys (|Z|≥|z|): **{prob_txt}** | % ajasta ±1σ: **{pct_txt}**")
