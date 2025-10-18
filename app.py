# app.py — lukee ticket_base.csv ja tarjoaa filtterit

import math
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Osakekuvaaja", layout="centered")

# -------------------------------------------
# 1) Lataa CSV (tuetaan sekä ticket_base.csv että tickers_base.csv)
# -------------------------------------------
root = Path(__file__).parent
csv_candidates = [root / "ticket_base.csv", root / "tickers_base.csv"]
df = None
for p in csv_candidates:
    if p.exists():
        df = pd.read_csv(p)
        break

# Varmista odotetut sarakkeet
needed_cols = {"symbol","name","exchange","country","asset_class","currency","notes"}
if df is None or not needed_cols.issubset(set(df.columns)):
    st.warning("Ticker-lista puuttuu tai sarakkeet eivät täsmää. "
               "Lisää repoosi juureen tiedosto **ticket_base.csv** yllä kuvatulla rakenteella. "
               "Käytetään pientä oletuslistaa.")
    df = pd.DataFrame({
        "symbol":["NOKIA.HE","SAMPO.HE","ELISA.HE","AAPL","MSFT","SPY"],
        "name":["Nokia","Sampo","Elisa","Apple","Microsoft","SPDR S&P 500 ETF"],
        "exchange":["Nasdaq Helsinki","Nasdaq Helsinki","Nasdaq Helsinki","NASDAQ","NASDAQ","NYSE Arca"],
        "country":["FI","FI","FI","US","US","US"],
        "asset_class":["Equity","Equity","Equity","Equity","Equity","ETF"],
        "currency":["EUR","EUR","EUR","USD","USD","USD"],
        "notes":["","","","","",""]
    })

# Siivoa/varmistuksia
df["name"] = df["name"].fillna("")
df["exchange"] = df["exchange"].fillna("")
df["country"] = df["country"].fillna("")
df["asset_class"] = df["asset_class"].fillna("")
df["currency"] = df["currency"].fillna("")
df["notes"] = df["notes"].fillna("")

# Rakennetaan label -> symbol -mapping (format_func hoitaa näytön, value=varsinainen symboli)
def make_label(row):
    extra = ", ".join([v for v in [row["exchange"], row["country"], row["currency"], row["asset_class"]] if v])
    return f'{row["symbol"]} — {row["name"]}' + (f" ({extra})" if extra else "")

df["label"] = df.apply(make_label, axis=1)

# -------------------------------------------
# 2) UI: asset-filtteri + symbolivalinta + aikajakso
# -------------------------------------------
years_options = {"1v":1, "5v":5, "10v":10, "15v":15}

st.title("📈 Yksinkertainen osakekuvaaja")
st.write("Valitse listasta tai kirjoita oma ticker. Data haetaan Yahoo Financesta ja kuvataan keskihinta sekä ±1/2/3σ-rajat. "
         "Otsikon alla näkyy myös poikkeaman kahden hännän todennäköisyys.")

col0, _ = st.columns([1,3])
asset_sel = col0.selectbox(
    "Asset class",
    options=["Kaikki"] + sorted(df["asset_class"].dropna().unique().tolist()),
    index=0
)

df_view = df if asset_sel == "Kaikki" else df[df["asset_class"] == asset_sel]
# Järjestetään aakkosittain labelin mukaan
df_view = df_view.sort_values("label")

col1, col2 = st.columns([2,1])
symbol = col1.selectbox(
    "Symboli (kirjoita hakeaksesi)",
    options=df_view["symbol"].tolist(),
    index=0 if not df_view.empty else None,
    format_func=lambda s: df_view.loc[df_view["symbol"]==s,"label"].iloc[0] if s in df_view["symbol"].values else s,
    placeholder="Esim. NOKIA.HE, SXR8.DE, ^GSPC"
)
custom = col1.text_input("…tai anna oma ticker (Enter)")
years_label = col2.selectbox("Aikajakso", list(years_options.keys()), index=0)

symbol = (custom.strip().upper() if custom.strip() else symbol).strip()
years = years_options[years_label]

# -------------------------------------------
# 3) Piirto & tilastot
# -------------------------------------------
def norm_cdf(x: float) -> float:
    # N(0,1) cdf ilman SciPyä
    return 0.5*(1.0 + math.erf(x/math.sqrt(2.0)))

def fetch_series(sym: str, years: int) -> pd.Series:
    """Lataa päivädataa ja palauta ensisijaisesti Adj Close, toissijaisesti Close."""
    data = yf.download(sym, period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
    if data is None or data.empty:
        return pd.Series(dtype=float)
    s = data["Adj Close"] if "Adj Close" in data.columns else data.get("Close")
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    return s.dropna()

if st.button("Näytä kuvaaja", type="primary"):
    s = fetch_series(symbol, years)
    if s.empty:
        st.warning(f"Ei saatavilla dataa: {symbol}")
        st.stop()

    mean = float(s.mean())
    std  = float(s.std(ddof=1)) if s.std(ddof=1)!=0 else 0.0
    last = float(s.iloc[-1])

    up1, up2, up3 = mean+std, mean+2*std, mean+3*std
    lo1, lo2, lo3 = mean-std, mean-2*std, mean-3*std

    if std==0:
        prob_tail = float("nan")
        pct1 = float("nan")
    else:
        z = (last-mean)/std
        prob_tail = 2*(1 - norm_cdf(abs(z)))     # kahden hännän todennäköisyys |Z|≥|z|
        pct1 = (s.between(mean-std, mean+std)).mean()*100

    fig, ax = plt.subplots(figsize=(10,5), dpi=120)
    ax.plot(s.index, s.values, label=f"{symbol} Adj Close", lw=1.6)
    ax.axhline(mean, color='green', ls='--', lw=1.2, label='Keskihinta')
    for y, ls in [(up1,'--'),(up2,'--'),(up3,'--'), (lo1,':'),(lo2,':'),(lo3,':')]:
        ax.axhline(y, ls=ls, lw=1)
    ax.scatter(s.index[-1], last, zorder=5)
    ax.set_title(f"{symbol} — {years_label}")
    ax.set_xlabel("Päivämäärä"); ax.set_ylabel("Hinta"); ax.grid(alpha=0.25); ax.legend(fontsize=9)
    st.pyplot(fig, clear_figure=True)

    prob_txt = "—" if not np.isfinite(prob_tail) else f"{prob_tail*100:.3f}%"
    pct_txt  = "—" if not np.isfinite(pct1) else f"{pct1:.1f}%"
    st.info(f"Poikkeaman todennäköisyys (|Z|≥|z|): **{prob_txt}** | % ajasta ±1σ: **{pct_txt}**")
