import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from pathlib import Path
import time
import requests
import warnings

# ------------------------------
# A) SESSION (User-Agent + requests)
# ------------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
})

# ------------------------------
# CSV lukeminen, UI setup (täysin kuten ennen)
# ------------------------------
st.set_page_config(page_title="Osakekuvaaja", layout="centered")
# ... (kaikki CSV-lukua ja valikoita koskevat osat tähän väliin)
# symbol, custom, years jne kuten nyt

# ------------------------------
# B) symbol_variants – lisää heti CSV/valintojen jälkeen
# ------------------------------
def symbol_variants(sym: str) -> list[str]:
    s = sym.strip().upper()
    out = [s]
    base = s.split(".")[0]

    if s.endswith(".DE"):
        out += [f"{base}.F"]
        if base == "ADS":
            out += ["ADDYY"]
    if s.endswith(".HE"):
        he_map = {
            "NOKIA": ["NOK"],
            "ELISA": ["ELMUF", "ELMAY"],
            "SAMPO": ["SAXPY"],
            "KNEBV": ["KNYJF"],
            "NESTE": ["NTOIY", "NTOIF"],
            "FORTUM": ["FOJCF"],
            "UPM": ["UPMKY"],
            "METSB": ["MTSAF"],
            "KESKOB": ["KKOYF", "KKOYB"],
            "WRT1V": ["WRTBY"]
        }
        if base in he_map:
            out += he_map[base]

    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

# ------------------------------
# C) fetch_series (tämä osa tulee B:n jälkeen)
# ------------------------------
@st.cache_data(ttl=600)
def fetch_series(sym: str, years: int) -> tuple[pd.Series, str | None, list[str]]:
    candidates = symbol_variants(sym)
    tried = []
    intervals = ["1d", "1wk"]
    periods = [years, years + 1]

    for cand in candidates:
        for per in periods:
            for itv in intervals:
                tried.append(f"{cand} [{per}y {itv}]")

                # 1) download
                try:
                    df = yf.download(
                        cand, period=f"{per}y", interval=itv,
                        auto_adjust=False, progress=False, threads=False,
                        session=SESSION,
                    )
                    if df is not None and not df.empty:
                        s = df["Adj Close"] if "Adj Close" in df.columns else df.get("Close")
                        if isinstance(s, pd.DataFrame):
                            s = s.squeeze()
                        s = s.dropna()
                        if not s.empty:
                            s.index = pd.to_datetime(s.index)
                            return s, cand, tried
                except Exception:
                    time.sleep(0.8)

                # 2) Ticker.history fallback
                try:
                    t = yf.Ticker(cand, session=SESSION)
                    hist = t.history(period=f"{per}y", interval=itv, auto_adjust=False)
                    if hist is not None and not hist.empty:
                        s = hist["Adj Close"] if "Adj Close" in hist.columns else hist.get("Close")
                        if isinstance(s, pd.DataFrame):
                            s = s.squeeze()
                        s = s.dropna()
                        if not s.empty:
                            s.index = pd.to_datetime(s.index)
                            return s, cand, tried
                except Exception:
                    time.sleep(0.8)

    return pd.Series(dtype=float), None, tried

# ------------------------------
# D) napin käsittely (UI-toiminto)
# ------------------------------
if st.button("Näytä kuvaaja", type="primary"):
    s, used, tried = fetch_series(symbol, years)
    if s.empty:
        st.warning(
            "Ei saatavilla dataa: **{}**.\n\n"
            "Yritetyt vaihtoehdot:\n- {}\n\n"
            "Vinkit: kokeile toista markkinapäätettä (esim. `.F` Saksaan) "
            "tai ADR:ää (esim. `NOK`)."
            .format(symbol, "\n- ".join(tried))
        )
        st.stop()

    # ... (mean/std/laskenta ja st.line_chart kuten ennen)
