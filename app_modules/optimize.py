# app_modules/optimize.py
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

def expected_annual_returns(returns: pd.DataFrame) -> pd.Series:
    mu = returns.mean()
    return (1 + mu) ** TRADING_DAYS_PER_YEAR - 1

def covariance_annual(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * TRADING_DAYS_PER_YEAR

def optimize_min_variance(mu: pd.Series, cov: pd.DataFrame, allow_short: bool=True):
    import cvxpy as cp
    n = len(mu)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if not allow_short:
        cons += [w >= 0]
    obj = cp.Minimize(cp.quad_form(w, cov.values))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        raise RuntimeError("Optimizasyon başarısız; kısıtları gevşetin.")
    weights = pd.Series(w.value.flatten(), index=mu.index)
    pmu = float(mu @ weights)
    psig = float(np.sqrt(weights.T @ cov.values @ weights))
    return weights, pmu, psig
