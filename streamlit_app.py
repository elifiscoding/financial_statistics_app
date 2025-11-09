# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from app_modules.data import fetch_yahoo_closes, products_supported, sectors_suggested, normalize_symbols_table
from app_modules.utils import momentum_score, volatility_score, format_pct
from app_modules.optimize import expected_annual_returns, covariance_annual, optimize_min_variance
from app_modules.options import option_recommendations_for_universe

TRADING_DAYS_PER_YEAR = 252
WEEK_DAYS = 5

st.set_page_config(layout="wide", page_title="6 Haftalık Portföy Optimizasyonu")
st.title("📈 6 Haftalık Portföy Maksimizasyonu — Gelişmiş Model")
st.caption("Yalnızca analiz — emir göndermez.")

st.subheader("1) Ticker, Ürün, Sektör ve Bütçe")
default_df = pd.DataFrame({
    "Ticker": ["AAPL","NVDA","TSLA","EURUSD","ES"],
    "Ürün":   ["Stocks","Stocks","Stocks","FX","Futures"],
    "Sektör": ["IT","AI","Elektrikli Arabalar","İletişim","Endüstri"]
})
user_df = st.data_editor(
    default_df, num_rows="dynamic", use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="Örn: AAPL, TSLA, EURUSD, ES, CL"),
        "Ürün":   st.column_config.SelectboxColumn("Ürün", options=products_supported),
        "Sektör": st.column_config.SelectboxColumn("Sektör", options=sectors_suggested)
    }
)
budget = st.number_input("Toplam Bütçe (USD)", value=100000, step=1000)
allow_short = st.checkbox("Short'a izin ver (negatif ağırlık)", value=True)

if st.button("Hesapla (6 Haftalık Model)"):
    # 2) Ürün bazlı sembol eşleştirme ve Yahoo destek kontrolü
    map_df, supported = normalize_symbols_table(user_df)
    st.subheader("2) Ürün Bazlı Sembol Eşleştirme")
    st.dataframe(map_df, use_container_width=True)
    if not supported:
        st.error("Desteklenen sembol yok (OK satırı bulunamadı).")
        st.stop()

    # 3) Veri -> 50 iş günü kapanış
    closes = fetch_yahoo_closes(supported)
    returns = closes.pct_change().dropna()

    st.subheader("3) Momentum ve Volatilite")
    mom = momentum_score(closes)
    vol = volatility_score(returns)
    stats = pd.DataFrame({"Momentum (50g)": mom, "Volatilite (yıllık)": vol})
    st.dataframe(stats.style.format("{:.2%}"), use_container_width=True)
    st.plotly_chart(px.imshow(stats.corr(), text_auto=True, title="Korelasyon (Momentum/Vol)"), use_container_width=True)

    # 4) Optimizasyon (Min Varyans)
    st.subheader("4) Optimizasyon")
    mu = expected_annual_returns(returns)
    cov = covariance_annual(returns)
    w, pmu, psig = optimize_min_variance(mu, cov, allow_short=allow_short)
    weights_df = pd.DataFrame({"Ağırlık": w}).T.T
    st.dataframe(weights_df.style.format("{:.2%}"), use_container_width=True)
    st.metric("Beklenen Yıllık Getiri", f"{pmu:.2%}")
    st.metric("Yıllık Volatilite", f"{psig:.2%}")

    # 5) 6 Haftalık Basit Simülasyon
    st.subheader("5) 6 Haftalık Beklenen Getiri (Basit)")
    weekly_return_est = pmu / (TRADING_DAYS_PER_YEAR / WEEK_DAYS)
    total_6w = (1 + weekly_return_est) ** 6 - 1
    st.metric("6 Haftalık Beklenen Getiri", f"{total_6w:.2%}")

    # 6) Opsiyon Analizi & Enstrüman Önerileri (hisse/ETF için)
    st.subheader("6) Opsiyon Analizi ve Enstrüman Önerileri")
    sector_map = {row["Yahoo"]: row["Sektör"] for _, row in map_df[map_df["Durum"]=="OK"].iterrows()}
    opt_df, recs = option_recommendations_for_universe(
        underlying_prices=closes.iloc[-1],
        momentum=mom,
        budget=budget,
        horizon_weeks=6
    )
    st.dataframe(opt_df, use_container_width=True)

    # 7) Ultra Basit “NE ALMALIYIM?” — bütçeye göre adet öner
    st.subheader("🔥 7) SONUÇ — Ne Almalıyım?")
    # Basit kural: ağırlık * bütçe -> ana enstrüman (hisse/ETF). 
    # Eğer aynı sembol için opsiyon önerisi 'daha iyi' ise, opsiyon tercih edilir.
    order_lines = []
    for sym, weight in w.sort_values(ascending=False).items():
        if weight <= 0:
            continue
        notional = float(budget * max(weight,0))
        suggestion = recs.get(sym, {})
        if suggestion.get("prefer") == "option":
            px_est = suggestion.get("price", np.nan)
            qty = int(notional // max(px_est,1e-6)) if np.isfinite(px_est) and px_est>0 else 0
            line = f"- {sym} (OPSİYON {suggestion.get('strategy')}): ~{qty} kontrat (@ ≈ ${px_est:.2f})"
        else:
            # Hisse/ETF
            px = float(closes.iloc[-1][sym])
            qty = int(notional // px) if px>0 else 0
            line = f"- {sym}: ~{qty} adet (@ ≈ ${px:.2f})"
        order_lines.append(line)
    lines = "\n".join(order_lines[:10])
    st.success(f"""
**6 haftalık stratejiye göre ultra basit alış listesi (yaklaşık):**

{lines}

> Not: Bu öneriler yalnızca eğitim amaçlıdır; işlem maliyetleri/döviz etkisi/likidite dikkate alınmamıştır.
""")
