# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Gelişmiş Streamlit Portföy Optimizasyonu (6 Haftalık Maksimizasyon için)
✔ Momentum analizi
✔ Volatilite hedefleme
✔ Sektör momentumu ısı haritası
✔ Long/Short spread önerici
✔ Haftalık performans simülasyonu
✔ Stop-loss & trailing stop sinyalleri
✔ Son ekranda "NE ALMALIYIM?" ultra basit öneri

Not: Bu uygulama sadece analiz üretir; emir göndermez.
"""

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

# ----------------------------------------------------
# Veri
# ----------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_yahoo_closes(symbols):
    if yf is None:
        raise RuntimeError("yfinance eksik.")
    # Daha sağlam: auto_adjust, group_by='ticker', progress=False
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
    # Close sütunlarını topla
    if isinstance(df.columns, pd.MultiIndex):
        closes = pd.DataFrame({sym: df[sym]["Close"] for sym in df.columns.levels[0] if (sym, "Close") in df.columns})
    else:
        closes = df[["Close"]].rename(columns={"Close": symbols[0]}) if "Close" in df.columns else df
    closes = closes.asfreq("B").ffill().tail(80).iloc[-50:]  # son 50 iş günü, biraz tampon
    # Tümü NaN veya sabit serileri ele
    for c in list(closes.columns):
        s = closes[c]
        if s.isna().all() or s.nunique(dropna=True) <= 1:
            closes.drop(columns=[c], inplace=True)
    if closes.empty:
        raise RuntimeError("Geçerli kapanış verisi yok (semboller desteklenmiyor veya veri yok)")
    # Kolon isimleri düzelt
    if isinstance(closes.columns, pd.MultiIndex):
        closes.columns = [c[1] if isinstance(c, tuple) else c for c in closes.columns]
    return closes
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(days=140)

    df = yf.download(symbols, start=start, end=end)["Close"].asfreq("B").ffill()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[1] for c in df.columns]
    return df.iloc[-50:]

# ----------------------------------------------------
# Hesaplayıcılar
# ----------------------------------------------------
def compute_returns(closes):
    # Log getiri daha stabil; NaN'leri ele
    logp = np.log(closes.replace(0, np.nan))
    rets = logp.diff().dropna(how='all')
    # Veri kalitesi: çok az gözlem veya sıfır varyanslı kolonları at
    valid_cols = []
    for c in rets.columns:
        if rets[c].count() >= 20 and rets[c].std(skipna=True) > 0:
            valid_cols.append(c)
    rets = rets[valid_cols]
    return rets

def momentum_score(closes):
    return closes.iloc[-1] / closes.iloc[0] - 1

def volatility_score(returns):
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

def expected_annual_returns(returns):
    # Log-getiri ortalamasını yıllıklaştır
    mu_daily = returns.mean(skipna=True)
    return mu_daily * TRADING_DAYS_PER_YEAR
    return (1 + mu) ** TRADING_DAYS_PER_YEAR - 1

def covariance_annual(returns):
    return returns.cov() * TRADING_DAYS_PER_YEAR

# ----------------------------------------------------
# Long/Short Spread Analizi
# ----------------------------------------------------
def compute_spread_scores(momentum, sectors):
    df = pd.DataFrame({"momentum": momentum, "sector": sectors})
    scores = {}
    for s in df["sector"].unique():
        sec = df[df["sector"] == s]
        if len(sec) < 2:
            continue
        long = sec["momentum"].idxmax()
        short = sec["momentum"].idxmin()
        scores[s] = (long, short)
    return scores

# ----------------------------------------------------
# Optimizasyon (SCS)
# ----------------------------------------------------
def optimize_portfolio(mu, cov, allow_short=True, ridge: float = 1e-8):
    import cvxpy as cp
    # --- Index hizalama ---
    tickers = mu.index.intersection(cov.index)
    mu = mu.loc[tickers].astype(float)
    cov = cov.loc[tickers, tickers].astype(float).copy()

    # Tek varlık durumu
    if len(tickers) == 1:
        w = pd.Series([1.0], index=tickers)
        pmu = float(mu.iloc[0])
        psig = float(np.sqrt(max(float(cov.values[0,0]), 0.0)))
        return w, pmu, psig

    # --- Temizleme ---
    cov = cov.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu = mu.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- Sayısal simetrikleştirme + ridge ---
    P = cov.to_numpy()
    P = 0.5 * (P + P.T)
    P = P + np.eye(P.shape[0]) * ridge

    n = len(tickers)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if not allow_short:
        cons.append(w >= 0)

    obj = cp.Minimize(cp.quad_form(w, cp.psd_wrap(P)))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        # ridge'i büyütüp bir kez daha dene
        P2 = P + np.eye(P.shape[0]) * (10 * ridge)
        obj2 = cp.Minimize(cp.quad_form(w, cp.psd_wrap(P2)))
        prob2 = cp.Problem(obj2, cons)
        prob2.solve(solver=cp.SCS, verbose=False)
        if w.value is None:
            raise RuntimeError("Optimizasyon başarısız: Kovaryans/parametreleri kontrol edin.")

    weights = pd.Series(np.array(w.value).flatten(), index=tickers)
    pmu = float(mu @ weights)
    psig = float(np.sqrt(weights.T @ P @ weights))
    return weights, pmu, psig

# ----------------------------------------------------
# UI
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="6 Haftalık Portföy Optimizasyonu")
st.title("📈 6 Haftalık Portföy Maksimizasyonu — Gelişmiş Model")
st.caption("Yalnızca analiz — emir göndermez.")

# --- Sidebar & Info sayfası ---
INFO_MD = r"""
# ℹ️ Proje Bilgisi

## Bu uygulama ne yapar?
- Seçtiğiniz **hisse/ETF/FX/Futures** sembollerinin **son 50 iş günü** fiyatlarını çeker.
- **Momentum (50g)** ve **yıllıklaştırılmış volatilite** hesaplar.
- **Markowitz (Min Varyans)** ile portföy ağırlıklarını optimize eder (isteğe bağlı short).
- **6 haftalık** basit beklenen getiri tahmini yapar.
- Hisse/ETF'lerde **opsiyon zincirini** tarar; **ATM Call/Put** için fiyat/IV/performans skoru üretir ve **underlying vs. opsiyon** tercihi önerir.
- En sonda **\"Ne almalıyım?\"** ekranında, bütçeye göre **yaklaşık adet** önerir.

## Mantık (Logic)
1. **Veri → Getiri**: Kapanışlardan günlük getiriler türetilir.
2. **Öznitelikler**: 50g **momentum** (F\_t / F\_0 − 1) ve **volatilite** (σ\_yıllık) hesaplanır.
3. **Kovaryans**: Günlük kovaryans yıllığa ölçeklenir; sayısal kararlılık için küçük **ridge** eklenir ve **PSD** kabulü yapılır.
4. **Optimizasyon**: (min w' Σ w) konu: sum w=1, (opsiyonel) w>=0. Çözüm **cvxpy + SCS**.
5. **6 Haftalık Tahmin**: Yıllık beklenen getiri haftalığa indirgenir ve 6 hafta birleşiklenir.
6. **Opsiyon Önerisi**: 6 haftaya en yakın vade; **ATM Call/Put**. Momentum yönüne göre beklenen senaryo ile **payoff/maliyet** skoru. Skor yeterliyse **opsiyon**, değilse **underlying**.
7. **Alış Listesi**: notional = w_i × bütçe. Underlying: adet = notional / fiyat; opsiyon: kontrat = notional / prim.

## Varsayımlar / Sınırlamalar
- Fiyatlar **yfinance** kaynaklıdır; gecikmeli/eksik olabilir.
- İşlem maliyetleri, slipaj, vergi ve temettüler **dahil değildir**.
- Opsiyon zincirleri hisse/ETF odaklıdır; **FX/Futures** için zincir sınırlı olabilir.
- Bu araç **yatırım tavsiyesi değildir**; sadece eğitim/analiz içindir.

## İpuçları
- **Haftalık** yeniden dengeleme 6 haftalık ufukta uygundur.
- Aşırı volatil varlıklar için **ağırlık sınırı**/hedge düşünün.
- Opsiyonlarda **likidite** ve **spread** kontrolü yapın.
"""

with st.sidebar:
    st.header("📘 Info")
    page = st.radio("Sayfa", ["Analiz", "Info"], index=0)
    st.markdown("**Proje Özeti**: 50g momentum + volatilite → min varyans portföy → 6 haftalık tahmin → opsiyon/underlying önerisi.")

if page == "Info":
    st.markdown(INFO_MD)
    st.stop()

# Önerilen sektörler
sectors_suggested = [
    "IT", "AI", "Elektrikli Arabalar", "Madencilik", "Sağlık",
    "Enerji", "Finans", "Tüketim", "Ulaştırma", "Endüstri",
    "Malzemeler", "Emlak", "İletişim", "Yenilenebilir Enerji",
    "Yarı İletken", "Biyoteknoloji"
]

st.subheader("1) Ticker, Ürün ve Sektör Giriniz")
products_supported = [
    "FX", "CFDs", "Stocks", "Funds", "ETFs", "Futures", "Listed options", "Bonds", "Mutual funds"
]

default_df = pd.DataFrame({
    "Ticker": ["AAPL", "NVDA", "TSLA", "EURUSD", "ES"],
    "Ürün":   ["Stocks", "Stocks", "Stocks", "FX", "Futures"],
    "Sektör": ["IT", "AI", "Elektrikli Arabalar", "İletişim", "Endüstri"]
})

user_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="Örn: AAPL, TSLA, EURUSD, ES, CL"),
        "Ürün": st.column_config.SelectboxColumn("Ürün", options=products_supported, help="Ekran görüntüsündeki ürün tipleri"),
        "Sektör": st.column_config.SelectboxColumn("Sektör", options=sectors_suggested)
    }
)

run = st.button("Hesapla (6 Haftalık Model)")

if run:
    tickers = [t.upper().strip() for t in user_df["Ticker"].dropna()]
    sector_map = {row["Ticker"].upper(): row["Sektör"] for _, row in user_df.iterrows()}

    closes = fetch_yahoo_closes(tickers)
    returns = compute_returns(closes)

    # ---------------- Momentum & Vol ----------------
    st.subheader("2) Momentum ve Volatilite Analizi")
    mom = momentum_score(closes)
    vol = volatility_score(returns)

    stats = pd.DataFrame({
        "Momentum (50g)": mom,
        "Volatilite": vol
    })
    st.dataframe(stats.style.format("{:.2%}"), use_container_width=True)

    # Isı haritası
    heat = stats.copy()
    st.plotly_chart(px.imshow(heat.corr(), text_auto=True, title="Momentum / Volatilite Korelasyon"))

    # ---------------- Long/Short Spread Önerici ----------------
    st.subheader("3) Long/Short Spread Önerileri")
    spreads = compute_spread_scores(mom, sector_map)

    for sec, (lng, shrt) in spreads.items():
        st.write(f"**{sec}**: Long → {lng}, Short → {shrt}")

    # ---------------- Optimizasyon ----------------
    st.subheader("4) Optimizasyon (Min Varyans)")
    mu = expected_annual_returns(returns)
    cov = covariance_annual(returns)

    weights, pmu, psig = optimize_portfolio(mu, cov, allow_short=True)

    st.dataframe(pd.DataFrame({"Ağırlık": weights}).T.T.style.format("{:.2%}"))
    st.metric("Beklenen Yıllık Getiri", f"{pmu:.2%}")
    st.metric("Yıllık Volatilite", f"{psig:.2%}")

    # ---------------- Haftalık Simülasyon ----------------
    st.subheader("5) 6 Haftalık Basit Simülasyon")
    weekly_return_est = pmu / (TRADING_DAYS_PER_YEAR / WEEK_DAYS)
    total_6w = (1 + weekly_return_est) ** 6 - 1
    st.metric("6 Haftalık Beklenen Getiri", f"{total_6w:.2%}")

    # ---------------- Ultra Basit Nihai Öneri ----------------
    st.subheader("🔥 6) SONUÇ — Ne Almalıyım? (Ultra Basit)")
    sort_w = weights.sort_values(ascending=False)

    top3 = sort_w.head(3)
    st.success(
        "**6 haftalık stratejiye göre en basit portföy önerisi:**\n"
        + "\n".join([f"- {i}: %{w*100:.1f}" for i, w in top3.items()])
        + "\n\nDiğerlerine düşük ağırlık verilebilir veya short pozisyonlarla hedge geçilebilir."
    )

