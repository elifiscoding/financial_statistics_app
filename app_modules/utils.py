# app_modules/utils.py
import pandas as pd
import numpy as np
TRADING_DAYS_PER_YEAR = 252

def momentum_score(closes: pd.DataFrame):
    return closes.iloc[-1] / closes.iloc[0] - 1

def volatility_score(returns: pd.DataFrame):
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

def format_pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "-"
