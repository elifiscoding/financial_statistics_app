# app_modules/options.py
import pandas as pd
import numpy as np
import yfinance as yf
from math import erf, sqrt, log, exp
from datetime import datetime, timedelta

def _norm_cdf(x):  # standard normal CDF
    return 0.5*(1.0+erf(x/sqrt(2)))

def _bs_call_price(S,K,r,sigma,T):
    if sigma <= 0 or T<=0 or S<=0 or K<=0: 
        return max(S-K,0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*_norm_cdf(d1) - K*np.exp(-r*T)*_norm_cdf(d2)

def _implied_vol_from_price_call(S,K,r,T,price,guess=0.3):
    # simple Newton-Raphson
    sigma = max(guess,1e-6)
    for _ in range(30):
        if sigma<=0: sigma=1e-6
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        vega = S * (1/np.sqrt(2*np.pi)) * np.exp(-0.5*d1**2) * sqrt(T)
        model = S*_norm_cdf(d1) - K*np.exp(-r*T)*_norm_cdf(d2)
        diff = model - price
        if abs(diff) < 1e-4 or vega < 1e-8:
            break
        sigma = sigma - diff/vega
    return float(abs(sigma))

def _nearest_expiry_after_weeks(ticker: str, weeks: int):
    tk = yf.Ticker(ticker)
    exps = tk.options
    if not exps: 
        return None
    target = datetime.utcnow() + timedelta(weeks=weeks)
    # choose the first expiry after target (or last available)
    dates = [datetime.strptime(d,"%Y-%m-%d") for d in exps]
    dates.sort()
    for d in dates:
        if d >= target:
            return d.strftime("%Y-%m-%d")
    return dates[-1].strftime("%Y-%m-%d")

def option_recommendations_for_universe(underlying_prices: pd.Series, momentum: pd.Series, budget: float, horizon_weeks: int=6, r: float=0.0):
    """
    For each equity/ETF symbol, propose either:
    - Underlying shares
    - ATM Call (if momentum positive)
    - ATM Put (if momentum negative)
    Also compute simple score = expected payoff / cost
    """
    rows=[]
    recs={}
    for sym, S in underlying_prices.items():
        # try to get an expiry ~ horizon
        expiry = _nearest_expiry_after_weeks(sym, horizon_weeks)
        if expiry is None:
            rows.append({"Symbol": sym, "Type": "NoOptions", "Detail": "Opsiyon zinciri yok"})
            recs[sym] = {"prefer":"underlying"}
            continue
        tk = yf.Ticker(sym)
        try:
            chain = tk.option_chain(expiry)
        except Exception:
            rows.append({"Symbol": sym, "Type": "NoOptions", "Detail": "Zincir çekilemedi"})
            recs[sym] = {"prefer":"underlying"}
            continue
        calls = chain.calls
        puts = chain.puts
        if calls is None or puts is None or len(calls)==0 or len(puts)==0:
            rows.append({"Symbol": sym, "Type": "NoOptions", "Detail": "Boş zincir"})
            recs[sym] = {"prefer":"underlying"}
            continue

        # choose ATM strike
        calls["dist"] = (calls["strike"]-S).abs()
        puts["dist"]  = (puts["strike"]-S).abs()
        call_atm = calls.sort_values("dist").iloc[0]
        put_atm  = puts.sort_values("dist").iloc[0]

        # time to expiry in years
        from datetime import datetime
        T = max((datetime.strptime(expiry,"%Y-%m-%d") - datetime.utcnow()).days,1)/365.0

        # use lastPrice as option price, fallback to bid/ask mid
        def px(row):
            for c in ["lastPrice","lastTradePrice","lastPrice","mark","bid","ask"]:
                if c in row and pd.notna(row[c]) and row[c]>0:
                    return float(row[c]) if c not in ("bid","ask") else float((row.get("bid",0)+row.get("ask",0))/2)
            return np.nan

        c_price = px(call_atm)
        p_price = px(put_atm)

        # rough IV from call price (if not present)
        Kc = float(call_atm["strike"]); Kp=float(put_atm["strike"])
        ivc = _implied_vol_from_price_call(S,Kc,r,T,c_price) if np.isfinite(c_price) else np.nan
        # expected payoff heuristic based on momentum: 
        m = float(momentum.get(sym,0))
        up_scen = S*(1+max(m,0))  # if momentum+, expect at least that move
        dn_scen = S*(1+min(m,0))
        exp_payoff_call = max(up_scen-Kc,0)
        exp_payoff_put  = max(Kp-dn_scen,0)

        score_call = exp_payoff_call / max(c_price,1e-6) if np.isfinite(c_price) and c_price>0 else 0.0
        score_put  = exp_payoff_put  / max(p_price,1e-6) if np.isfinite(p_price) and p_price>0 else 0.0

        rows.append({"Symbol": sym, "Type": "ATM Call", "Expiry": expiry, "Strike": Kc, "Price": c_price, "IV_est": ivc, "Score": score_call})
        rows.append({"Symbol": sym, "Type": "ATM Put",  "Expiry": expiry, "Strike": Kp, "Price": p_price, "IV_est": None, "Score": score_put})

        # Decide preference
        prefer = "underlying"
        strat = "-"
        price = None
        if m > 0 and score_call > 1.0:
            prefer = "option"; strat = "ATM Call"; price = c_price
        elif m < 0 and score_put > 1.0:
            prefer = "option"; strat = "ATM Put"; price = p_price
        recs[sym] = {"prefer": prefer, "strategy": strat, "price": price}

    table = pd.DataFrame(rows)
    if not table.empty:
        # order by Score desc
        table = table.sort_values(["Symbol","Score"], ascending=[True, False])
    return table, recs
