# streamlit_app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import yfinance as yf
except:
    yf = None

TRADING_DAYS_PER_YEAR = 252
WEEK_DAYS = 5

# =======================================================================
# DATA FETCHER
# =======================================================================
@st.cache_data(show_spinner=False)
def fetch_yahoo_closes(symbols):
    if yf is None:
        raise RuntimeError("yfinance eksik.")

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(days=200)

    df = yf.download(
        symbols,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    # MultiIndex düzelt
    if isinstance(df.columns, pd.MultiIndex):
        closes = pd.DataFrame({sym: df[sym]["Close"] for sym in df.columns.levels[0] if ("Close" in df[sym])})
    else:
        closes = df

    closes = closes.asfreq("B").ffill().tail(80).iloc[-50:]

    # Bozuk / düz serileri ele
    for c in list(closes.columns):
        s = closes[c]
        if s.isna().all() or s.nunique(dropna=True) <= 1:
            closes.drop(columns=[c], inplace=True)

    if closes.empty:
        raise RuntimeError("Hiç düzgün fiyat verisi yok.")

    return closes


# =======================================================================
# BASIC CALCS
# =======================================================================
def compute_returns(closes):
    logp = np.log(closes.replace(0, np.nan))
    rets = logp.diff().dropna(how="all")
    valid = [c for c in rets.columns if rets[c].count() >= 20 and rets[c].std() > 0]
    return rets[valid]


def momentum_score(closes):
    return closes.iloc[-1] / closes.iloc[0] - 1


def volatility_score(returns):
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def expected_annual_returns(returns):
    mu_daily = returns.mean()
    return mu_daily * TRADING_DAYS_PER_YEAR


def covariance_annual(returns):
    return returns.cov() * TRADING_DAYS_PER_YEAR


# =======================================================================
# SPREAD ANALYSIS
# =======================================================================
def compute_spread_scores(momentum, sectors):
    df = pd.DataFrame({"momentum": momentum, "sector": sectors})
    out = {}
    for sec in df["sector"].unique():
        sub = df[df["sector"] == sec]
        if len(sub) < 2:
            continue
        out[sec] = (sub["momentum"].idxmax(), sub["momentum"].idxmin())
    return out


# =======================================================================
# PORTFOLIO OPTIMIZATION
# =======================================================================
def optimize_portfolio(mu, cov, allow_short=True, ridge=1e-8):
    import cvxpy as cp

    tickers = mu.index
    P = cov.to_numpy()

    # Symmetrize + ridge
    P = 0.5 * (P + P.T)
    P = P + np.eye(len(tickers)) * ridge

    w = cp.Variable(len(tickers))
    cons = [cp.sum(w) == 1]
    if not allow_short:
        cons.append(w >= 0)

    obj = cp.Minimize(cp.quad_form(w, cp.psd_wrap(P)))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        raise RuntimeError("Optimizasyon başarısız.")

    w = pd.Series(w.value, index=tickers)
    pmu = float(mu @ w)
    psig = float(np.sqrt(w.T @ P @ w))

    return w, pmu, psig


# =======================================================================
# STREAMLIT APP
# =======================================================================
st.set_page_config(page_title="6 Haftalık Portföy Optimizasyonu", layout="wide")
st.title("📈 6 Haftalık Portföy Maksimizasyonu — Gelişmiş Model")

if "results" not in st.session_state:
    st.session_state.results = None

# ------------------- SEKTÖR EVRENİ -------------------
sectors_suggested = [
    "IT","AI","Elektrikli Arabalar","Madencilik","Sağlık","Enerji","Finans","Tüketim",
    "Ulaştırma","Endüstri","Malzemeler","Emlak","İletişim",
    "Yenilenebilir Enerji","Yarı İletken","Biyoteknoloji"
]

sector_universe = {
    "IT": ["AAPL","MSFT","ORCL","CRM"],
    "AI": ["NVDA","AVGO","MSFT","GOOGL"],
    "Elektrikli Arabalar": ["TSLA","RIVN","NIO","LI"],
    "Madencilik": ["BHP","RIO","VALE","FCX"],
    "Sağlık": ["UNH","JNJ","MRK","PFE"],
    "Enerji": ["XOM","CVX","COP","SLB"],
    "Finans": ["JPM","BAC","C","GS"],
    "Endüstri": ["HON","CAT","DE","MMM"],
    "İletişim": ["META","GOOGL","NFLX","DIS"],
    "Yarı İletken": ["NVDA","AMD","TSM","INTC"],
}


# =======================================================================
# INPUT FORM
# =======================================================================
with st.form("run_form"):
    st.subheader("1) Sektör veya Ticker ile Evren Oluştur")

    mode = st.radio(
        "Evren oluşturma yöntemi:",
        ["Sektör seçerek otomatik oluştur", "Ticker’ları kendim gireceğim"],
    )

    if mode == "Sektör seçerek otomatik oluştur":
        selected = st.multiselect("Sektör seç", sectors_suggested, ["IT", "AI"])
        rows = []
        for sec in selected:
            for t in sector_universe.get(sec, []):
                rows.append({"Ticker": t, "Sektör": sec})
        df_input = pd.DataFrame(rows)

    else:
        df_input = pd.DataFrame({
            "Ticker": ["AAPL", "NVDA", "TSLA"],
            "Sektör": ["IT", "AI", "Elektrikli Arabalar"]
        })

    df_input = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

    run = st.form_submit_button("Hesapla (6 Haftalık Model)")

# =======================================================================
# RUN CALCULATION
# =======================================================================
if run:
    tickers = [t.upper().strip() for t in df_input["Ticker"]]
    sector_map = {row["Ticker"].upper(): row["Sektör"] for _, row in df_input.iterrows()}

    closes = fetch_yahoo_closes(tickers)
    returns = compute_returns(closes)
    mom = momentum_score(closes)
    vol = volatility_score(returns)
    mu = expected_annual_returns(returns)
    cov = covariance_annual(returns)
    weights, pmu, psig = optimize_portfolio(mu, cov, allow_short=True)

    st.session_state.results = {
        "tickers": tickers,
        "closes": closes,
        "returns": returns,
        "mom": mom,
        "vol": vol,
        "mu": mu,
        "cov": cov,
        "weights": weights,
        "pmu": pmu,
        "psig": psig,
        "sector_map": sector_map
    }

# =======================================================================
# DISPLAY RESULTS
# =======================================================================
res = st.session_state.results
if res is None:
    st.info("Portföyü görmek için 'Hesapla' butonuna basın.")
    st.stop()

# Unpack
closes = res["closes"]
returns = res["returns"]
mom = res["mom"]
vol = res["vol"]
weights = res["weights"]
pmu = res["pmu"]
psig = res["psig"]
sector_map = res["sector_map"]

# ===============================================================
# 2) Momentum – Vol
# ===============================================================
st.subheader("2) Momentum ve Volatilite")
st.dataframe(pd.DataFrame({"Momentum": mom, "Volatilite": vol}).style.format("{:.2%}"))
st.plotly_chart(px.imshow(pd.DataFrame({"Momentum": mom, "Vol": vol}).corr(),
                          text_auto=True, title="Korelasyon"))

# ===============================================================
# 3) Spread Önerileri
# ===============================================================
st.subheader("3) Long/Short Spread Önerileri")
spreads = compute_spread_scores(mom, sector_map)
for sec, (lng, shrt) in spreads.items():
    st.write(f"**{sec}**: LONG → {lng}, SHORT → {shrt}")

# ===============================================================
# 4) Optimizasyon Sonuçları
# ===============================================================
st.subheader("4) Optimizasyon: Min Varyans")
st.dataframe(pd.DataFrame({"Ağırlık": weights}).T.T.style.format("{:.2%}"))
st.metric("Beklenen Yıllık Getiri", f"{pmu:.2%}")
st.metric("Yıllık Volatilite", f"{psig:.2%}")

# ===============================================================
# 5) 6 Haftalık Basit Simülasyon
# ===============================================================
st.subheader("5) 6 Haftalık Basit Simülasyon")
weekly_return_est = pmu / (TRADING_DAYS_PER_YEAR / WEEK_DAYS)
total_6w = (1 + weekly_return_est) ** 6 - 1
st.metric("6 Haftalık Beklenen Getiri", f"{total_6w:.2%}")

# ===============================================================
# 6) OPTIMAL PORTFÖY + ALIŞ LİSTESİ
# ===============================================================
st.subheader("6) Optimal Portföy (Tam Liste)")
sort_w = weights.sort_values(ascending=False)

portdf = pd.DataFrame({
    "Ağırlık": sort_w,
    "Son Fiyat": closes.iloc[-1].reindex(sort_w.index)
})
st.dataframe(portdf.style.format({"Ağırlık": "{:.2%}", "Son Fiyat": "${:,.2f}"}), use_container_width=True)

st.plotly_chart(px.bar(sort_w, title="Optimal Portföy Ağırlıkları"), use_container_width=True)

# ------------------- Alım Listesi -------------------
st.subheader("🔥 7) Ne Almalıyım? (Adet Hesaplı)")
budget = st.number_input("Bütçe (USD)", value=100000, step=1000)
min_w = st.slider("Minimum ağırlık (%)", 0.0, 5.0, 0.5, 0.1)

lines = []
for t, w in sort_w.items():
    if w * 100 < min_w:
        continue

    price = closes.iloc[-1][t]
    notional = budget * w
    qty = int(notional // price)

    lines.append(f"- {t}: %{w*100:.1f} → {qty} adet @ ${price:,.2f}")

st.success("\n".join(lines) if lines else "Eşik çok yüksek olabilir.")
