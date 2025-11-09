# app_modules/optimize.py
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

def expected_annual_returns(returns: pd.DataFrame) -> pd.Series:
    mu = returns.mean()
    return (1 + mu) ** TRADING_DAYS_PER_YEAR - 1

def covariance_annual(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * TRADING_DAYS_PER_YEAR

def optimize_min_variance(mu: pd.Series,
                          cov: pd.DataFrame,
                          allow_short: bool = True,
                          ridge: float = 1e-8):
    import cvxpy as cp

    # 1) Index hizalama (aynı sırada ve kesişimde olsun)
    tickers = mu.index.intersection(cov.index)
    mu = mu.loc[tickers].astype(float)
    cov = cov.loc[tickers, tickers].astype(float).copy()

    # Tek varlık durumu: ağırlık 1
    if len(tickers) == 1:
        w = pd.Series([1.0], index=tickers)
        pmu = float(mu.iloc[0])
        psig = float(np.sqrt(max(cov.values[0, 0], 0.0)))
        return w, pmu, psig

    # 2) Temizle (NaN/Inf → 0)
    cov = cov.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu = mu.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 3) Sayısal simetrikleştirme + küçük ridge (PSD yakınsaması)
    P = cov.to_numpy()
    P = 0.5 * (P + P.T)
    P = P + np.eye(P.shape[0]) * ridge

    n = len(tickers)
    w = cp.Variable(n)

    cons = [cp.sum(w) == 1]
    if not allow_short:
        cons += [w >= 0]

    # 4) PSD sarmalayıcı ile quad_form
    obj = cp.Minimize(cp.quad_form(w, cp.psd_wrap(P)))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        # Bir tık daha büyük ridge ile bir deneme daha
        P2 = P + np.eye(P.shape[0]) * (10 * ridge)
        obj2 = cp.Minimize(cp.quad_form(w, cp.psd_wrap(P2)))
        prob2 = cp.Problem(obj2, cons)
        prob2.solve(solver=cp.SCS, verbose=False)

        if w.value is None:
            raise RuntimeError("Optimizasyon başarısız: Kovaryans/parametreleri kontrol edin.")

    weights = pd.Series(np.array(w.value).flatten(), index=tickers)

    # Metrikler (hizalanmış veri ile)
    pmu = float(mu @ weights)
    psig = float(np.sqrt(weights.T @ P @ weights))

    return weights, pmu, psig
