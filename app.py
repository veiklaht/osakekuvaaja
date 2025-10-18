import math
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Osakekuvaaja", layout="centered")

years_options = {"1v":1, "5v":5, "10v":10, "15v":15}
suggested = [
    "NOKIA.HE","SAMPO.HE","ELISA.HE","KNEBV.HE","NESTE.HE","FORTUM.HE",
    "AAPL","MSFT","AMZN","NVDA","SPY","QQQ","VWCE.DE","EUNL.DE","SXR8.DE"
]

def norm_cdf(x):  # ilman SciPyä
    return 0.5*(1.0 + math.erf(x/math.sqrt(2)))

st.title("📈 Yksinkertainen osakekuvaaja")
st.write("Valitse osake listasta tai kirjoita oma ticker. Sovellus hakee datan Yahoo Financesta ja piirtää hinnan + sigma-rajat.")

col1, col2 = st.columns([2,1])
ticker = col1.selectbox("Osake / ETF / Indeksi", suggested, index=0, placeholder="Esim. NOKIA.HE")
custom = col1.text_input("…tai kirjoita oma ticker (Enter)", value="")
years_label = col2.selectbox("Aikajakso", list(years_options.keys()), index=0)

symbol = (custom.strip().upper() if custom.strip() else ticker).strip()
years = years_options[years_label]

if st.button("Näytä kuvaaja", type="primary"):
    try:
        data = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
    except Exception as e:
        st.error(f"Virhe haussa: {e}")
        st.stop()

    if data.empty:
        st.warning(f"Ei saatavilla dataa: {symbol}")
        st.stop()

    price = data.get("Adj Close", data.get("Close"))
    if isinstance(price, pd.DataFrame): price = price.squeeze()
    price = price.dropna()
    if price.empty:
        st.warning("Ei riittävästi hintadataa.")
        st.stop()

    mean = float(price.mean())
    std  = float(price.std(ddof=1)) if price.std(ddof=1)!=0 else 0.0
    last = float(price.iloc[-1])

    up1, up2, up3 = mean+std, mean+2*std, mean+3*std
    lo1, lo2, lo3 = mean-std, mean-2*std, mean-3*std

    if std==0:
        prob_tail = float("nan")
        pct1 = float("nan")
    else:
        z = (last-mean)/std
        prob_tail = 2*(1-norm_cdf(abs(z)))          # |Z|≥|z|
        pct1 = (price.between(mean-std, mean+std)).mean()*100

    fig, ax = plt.subplots(figsize=(10,5), dpi=120)
    ax.plot(price.index, price.values, label=f"{symbol} Adj Close", lw=1.6)
    ax.axhline(mean, color='green', ls='--', lw=1.2, label='Keskihinta')
    for y,ls in [(up1,'--'),(up2,'--'),(up3,'--'),(lo1,':'),(lo2,':'),(lo3,':')]:
        ax.axhline(y, ls=ls, lw=1)
    ax.scatter(price.index[-1], last, zorder=5)
    ax.set_title(f"{symbol} — {years_label}")
    ax.set_xlabel("Päivämäärä"); ax.set_ylabel("Hinta"); ax.grid(alpha=0.25); ax.legend(fontsize=9)
    st.pyplot(fig, clear_figure=True)

    prob_txt = "—" if not np.isfinite(prob_tail) else f"{prob_tail*100:.3f}%"
    pct_txt  = "—" if not np.isfinite(pct1) else f"{pct1:.1f}%"
    st.info(f"Poikkeaman todennäköisyys (|Z|≥|z|): **{prob_txt}** | % ajasta ±1σ: **{pct_txt}**")
