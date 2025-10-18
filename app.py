# @title
# --- Six Sigma Stock Plot + CSV + hakukenttä (Twelve Data + matplotlib) ---
import os
import io
import math
import time
import json
import unicodedata
from functools import lru_cache
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.stats import norm
from ipywidgets import Dropdown, Text, Password, HBox, VBox, Layout, Button, Output, Label
from IPython.display import display

# =========================
# 0) ASETUKSET & API-AVAIN
# =========================
# Yritä lukea env:stä; voit myös täyttää UI:ssa
DEFAULT_TWELVE_KEY = os.environ.get("TWELVEDATA_API_KEY", "").strip()

# HTTP-sessio (selain-tyylinen UA + keep-alive)
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/121.0.0.0 Safari/537.36"),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
})

# =========================
# 1) SYMBOLILISTA CSV:stä
# =========================
CSV_PATHS = ["/content/tickers_base.csv", "tickers_base.csv", "ticket_base.csv"]
df_tickers = None
for p in CSV_PATHS:
    if os.path.exists(p):
        df_tickers = pd.read_csv(p)
        break

fallback_symbols = ['NOKIA.HE', 'SAMPO.HE', 'ELISA.HE', 'AMZN', 'AAPL']

# =========================
# 2) AIKAJAKSOT (label -> laskenta)
# =========================
period_options = {
    "1kk": dict(months=1),
    "3kk": dict(months=3),
    "6kk": dict(months=6),
    "YTD": "ytd",          # vuoden alusta
    "1v": dict(years=1),
    "5v": dict(years=5),
    "10v": dict(years=10),
    "15v": dict(years=15),
}

def compute_start_end(period_label: str):
    """Palauta (start_date, end_date) UTC päivätasolla."""
    end_dt = datetime.utcnow()
    if period_label == "YTD":
        start_dt = datetime(end_dt.year, 1, 1)
    else:
        spec = period_options[period_label]
        # karkea lasku: kuukauset ~30 päivää, vuodet ~365 päivää
        days = 0
        if "months" in spec:
            days += spec["months"] * 30
        if "years" in spec:
            days += spec["years"] * 365
        start_dt = end_dt - timedelta(days=days + 7)  # pieni puskuri
    # Twelve Data hyväksyy YYYY-MM-DD
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

# =========================
# 3) APUTYÖKALUT
# =========================
def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s))
                   if not unicodedata.combining(c)).lower()

_SUFFIX_TO_EXCHANGE = {
    "HE": "Helsinki",
    "ST": "Stockholm",
    "CO": "Copenhagen",
    "OL": "Oslo",
    "DE": "Germany (Xetra/Frankfurt)",
    "PA": "Paris",
    "AS": "Amsterdam",
    "MI": "Milan",
    "L":  "London",
    "SW": "Switzerland",
    "TO": "Canada (TSX)",
    "V":  "Canada (TSXV)",
    "AX": "Australia",
    "HK": "Hong Kong",
    "SI": "Singapore",
    "KS": "Korea (KSE)",
    "KQ": "KOSDAQ",
    "SA": "Brazil (B3)",
    "NZ": "New Zealand",
    "TA": "Tel Aviv",
    "SS": "Shanghai",
    "SZ": "Shenzhen",
    "BK": "Thailand",
    "JK": "Indonesia",
    "TW": "Taiwan",
    "T":  "Tokyo",
}

def infer_exchange_from_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        return "USA (NYSE/Nasdaq)"
    if "." in symbol:
        suf = symbol.rsplit(".", 1)[-1].upper()
        return _SUFFIX_TO_EXCHANGE.get(suf, f"Other (.{suf})")
    return "USA (NYSE/Nasdaq)"

def get_exchange_col(df: pd.DataFrame) -> str | None:
    for col in ["exchange", "mic", "MIC", "Exchange", "porssi", "market"]:
        if col in df.columns:
            return col
    return None

def available_exchanges() -> list[str]:
    ex_list = []
    if df_tickers is not None and "symbol" in df_tickers.columns:
        exch_col = get_exchange_col(df_tickers)
        if exch_col:
            raw = df_tickers[exch_col].dropna().astype(str).str.strip()
            # muunna MIC -> nimiä, jos tarve
            mic_map = {
                "XHEL": "Helsinki", "XSTO": "Stockholm", "XCSE": "Copenhagen", "XOSL": "Oslo",
                "XETR": "Germany (Xetra/Frankfurt)", "XFRA": "Germany (Frankfurt)",
                "XPAR": "Paris", "XAMS": "Amsterdam", "XMIL": "Milan",
                "XLON": "London", "XSWX": "Switzerland",
                "XTSE": "Canada (TSX)", "XTSX": "Canada (TSXV)",
                "XNAS": "USA (Nasdaq)", "XNYS": "USA (NYSE)", "XHKG": "Hong Kong", "XSES": "Singapore",
            }
            normed = raw.map(lambda x: mic_map.get(x.upper(), x))
            ex_list = sorted(set(normed.tolist()))
        else:
            ex_list = sorted(set(infer_exchange_from_symbol(s)
                                 for s in df_tickers["symbol"].dropna().astype(str)))
    else:
        ex_list = sorted(set(infer_exchange_from_symbol(s) for s in fallback_symbols))
    return ["Kaikki"] + ex_list

def build_full_options(asset_class="Kaikki", exchange_choice="Kaikki"):
    if df_tickers is None or "symbol" not in df_tickers.columns:
        symbols = fallback_symbols
        if exchange_choice != "Kaikki":
            symbols = [s for s in symbols if infer_exchange_from_symbol(s) == exchange_choice]
        return [(s, s) for s in symbols]

    df = df_tickers.copy()

    if asset_class != "Kaikki" and "asset_class" in df.columns:
        df = df[df["asset_class"] == asset_class]

    if exchange_choice != "Kaikki":
        exch_col = get_exchange_col(df)
        if exch_col:
            mask = df[exch_col].astype(str).str.lower().str.contains(exchange_choice.lower())
            df = df[mask]
        else:
            df = df[df["symbol"].astype(str).map(infer_exchange_from_symbol) == exchange_choice]

    if "name" in df.columns:
        labels = (df["name"].fillna(df["symbol"]) + " (" + df["symbol"] + ")")
    else:
        labels = df["symbol"]

    options = list(zip(labels.tolist(), df["symbol"].tolist()))
    # poista duplikaatit
    seen, dedup = set(), []
    for lab, val in options:
        if val not in seen:
            seen.add(val)
            dedup.append((lab, val))
    return sorted(dedup, key=lambda x: strip_accents(x[0]))

def filter_options(options, query: str):
    if not query.strip():
        return options
    q = strip_accents(query)
    out = [(lab, val) for lab, val in options
           if q in strip_accents(lab) or q in strip_accents(val)]
    return out if out else options

def label_to_symbol(label_or_symbol: str):
    txt = label_or_symbol.strip()
    if txt.endswith(")") and "(" in txt:
        return txt.split("(")[-1][:-1].strip()
    return txt

# =========================
# 4) TWELVE DATA HAKU
# =========================
def _td_time_series(symbol: str, start_date: str, end_date: str, interval="1day", api_key: str = "") -> pd.DataFrame | None:
    """
    Twelve Data time_series:
      - symbol: esim "NOKIA.HE", "AAPL"
      - interval: "1day" tai "1week"
      - start_date/end_date: "YYYY-MM-DD"
    Palauttaa DataFrame: index=Date, columns=[Close, Adj Close]
    """
    if not api_key:
        return None
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={symbol}"
        f"&interval={interval}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&order=ASC"
        f"&outputsize=5000"
        f"&apikey={api_key}"
    )
    # pieni throttle
    time.sleep(0.4)
    r = SESSION.get(url, timeout=20)
    if not r.ok:
        return None
    try:
        js = r.json()
    except Exception:
        return None
    if "values" not in js:
        return None
    df = pd.DataFrame(js["values"])
    if df.empty or "datetime" not in df.columns or "close" not in df.columns:
        return None
    df["Date"]  = pd.to_datetime(df["datetime"], errors="coerce")
    df["Close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["Date","Close"]).set_index("Date").sort_index()
    df["Adj Close"] = df["Close"]  # yhteensopivuus
    return df

@lru_cache(maxsize=512)
def td_series_cached(symbol: str, period_label: str, interval="1day", api_key="") -> pd.Series | None:
    """Kääre, joka palauttaa yksittäisen hinnan sarjana (Adj Close / Close)."""
    sd, ed = compute_start_end(period_label)
    df = _td_time_series(symbol, sd, ed, interval=interval, api_key=api_key)
    if df is None or df.empty:
        return None
    s = df["Adj Close"] if "Adj Close" in df.columns else df.get("Close")
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    return s.dropna() if s is not None else None

# =========================
# 5) PIIRTO
# =========================
def _plot_to_axes(ax, symbol: str, period_label: str, price: pd.Series):
    mean_price = float(price.mean())
    std_price  = float(price.std(ddof=1)) if price.std(ddof=1) != 0 else 0.0
    last_price = float(price.iloc[-1])

    if std_price == 0:
        prob_tail = np.nan
        pct_within = np.nan
    else:
        z_last = (last_price - mean_price) / std_price
        prob_tail = 2 * (1 - norm.cdf(abs(z_last)))
        pct_within = (price.between(mean_price - std_price, mean_price + std_price)).mean() * 100

    ax.plot(price.index, price.values, linewidth=1.5, label=f"{symbol}")
    ax.axhline(mean_price, linestyle='--', color='green', label='Mean')

    for k, col in zip([1, 2, 3], ['blue', 'orange', 'red']):
        ax.axhline(mean_price + k*std_price, linestyle='--', color=col)
        ax.axhline(mean_price - k*std_price, linestyle=':', color=col)

    ax.scatter(price.index[-1], last_price, zorder=5)
    if std_price == 0:
        title = f"{symbol} — {period_label}\nσ=0 (ei vaihtelua)"
    else:
        title = f"{symbol} — {period_label}\nP(|Z|≥|zₙ|): {(prob_tail*100):.3f}% | % ±1σ: {pct_within:.1f}%"

    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

# =========================
# 6) WIDGETIT & LOGIIKKA
# =========================
asset_classes = ["Kaikki"] + (sorted(df_tickers["asset_class"].dropna().unique().tolist())
                              if df_tickers is not None and "asset_class" in df_tickers.columns else [])
exchanges = available_exchanges()

asset_dd   = Dropdown(options=asset_classes, value="Kaikki",
                      description='Asset class:', layout=Layout(width="240px"))
exchange_dd = Dropdown(options=exchanges, value="Kaikki",
                       description='Pörssi:', layout=Layout(width="240px"))

all_opts   = build_full_options(asset_dd.value, exchange_dd.value)
sym_dd     = Dropdown(options=all_opts, description='Symboli:', layout=Layout(width="460px"))
search_txt = Text(value="", placeholder='Hae esim. "Metsä", "Nokia", "VWCE"...',
                  description='Haku:', layout=Layout(width="360px"))
period_dd  = Dropdown(options=list(period_options.keys()), value='1v', description='Aikajakso:')

api_label = Label("Twelve Data API Key:")
api_key_w = Password(value=DEFAULT_TWELVE_KEY, placeholder="Liitä avain tähän", layout=Layout(width="280px"))

_selected_plots = []
grid_out = Output()
msg_out  = Output()

def refresh_symbol_options(*_):
    base = build_full_options(asset_dd.value, exchange_dd.value)
    filtered = filter_options(base, search_txt.value)
    current_value = sym_dd.value
    sym_dd.options = filtered
    values = [v for _, v in filtered]
    if current_value not in values and values:
        sym_dd.value = values[0]

asset_dd.observe(lambda _: refresh_symbol_options(), names='value')
exchange_dd.observe(lambda _: refresh_symbol_options(), names='value')
search_txt.observe(lambda _: refresh_symbol_options(), names='value')
refresh_symbol_options()

add_btn   = Button(description="Lisää kuvaaja", button_style="success", layout=Layout(width="150px"))
clear_btn = Button(description="Tyhjennä kaikki", button_style="danger", layout=Layout(width="150px"))

def _render_grid():
    with grid_out:
        grid_out.clear_output(wait=True)
        n = len(_selected_plots)
        if n == 0:
            display(pd.DataFrame({"Vinkki": ["Syötä Twelve Data API key, valitse/hakemerkkaa ticker ja lisää kuvaaja."]}))
            return
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.2*rows), squeeze=False)
        apikey = api_key_w.value.strip()
        for idx, (sym, per_lbl) in enumerate(_selected_plots):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            s = td_series_cached(sym, per_lbl, interval="1day", api_key=apikey)
            if s is None or s.empty:
                ax.text(0.5, 0.5, f"Ei dataa (Twelve Data):\n{sym} ({per_lbl})",
                        ha="center", va="center")
                ax.axis("off")
                continue
            _plot_to_axes(ax, sym, per_lbl, s)
        for k in range(n, rows*cols):
            r, c = divmod(k, cols)
            axes[r][c].axis("off")
        fig.tight_layout()
        plt.show()

def _add_plot_clicked(_):
    with msg_out:
        msg_out.clear_output(wait=True)
        if not api_key_w.value.strip():
            print("⚠️ Lisää Twelve Data API key ensin.")
            return
    _selected_plots.append((label_to_symbol(sym_dd.value), period_dd.value))
    _render_grid()

def _clear_clicked(_):
    _selected_plots.clear()
    _render_grid()

add_btn.on_click(_add_plot_clicked)
clear_btn.on_click(_clear_clicked)

ui = VBox([
    HBox([api_label, api_key_w]),
    HBox([asset_dd, exchange_dd]),
    HBox([sym_dd, period_dd, add_btn, clear_btn]),
    HBox([search_txt]),
    msg_out,
    grid_out
])
display(ui)
_render_grid()
