# app_modules/data.py
import pandas as pd
import yfinance as yf

products_supported = [
    "FX", "CFDs", "Stocks", "Funds", "ETFs", "Futures", "Listed options", "Bonds", "Mutual funds"
]

sectors_suggested = [
    "IT","AI","Elektrikli Arabalar","Madencilik","Sağlık",
    "Enerji","Finans","Tüketim","Ulaştırma","Endüstri",
    "Malzemeler","Emlak","İletişim","Yenilenebilir Enerji",
    "Yarı İletken","Biyoteknoloji"
]

def _to_yahoo_symbol(ticker: str, product: str) -> tuple[str, str]:
    t = ticker.upper().replace(" ", "")
    if product == "FX":
        if not t.endswith("=X"):
            t = t + "=X"
        return t, "OK"
    if product == "Futures":
        if not t.endswith("=F"):
            t = t + "=F"
        return t, "OK"
    if product in {"Stocks","ETFs","Funds","Mutual funds"}:
        return t, "OK"
    if product == "Bonds":
        return t, "ETF proxy önerilir (TLT/IEF/HYG)"
    if product in {"CFDs","Listed options"}:
        return t, "Altta yatanla analiz"
    return t, "Bilinmeyen ürün"

def normalize_symbols_table(df: pd.DataFrame):
    rows=[]
    for _, r in df.dropna(subset=["Ticker","Ürün"]).iterrows():
        ysym, note = _to_yahoo_symbol(r["Ticker"], r["Ürün"])
        rows.append({
            "Girdi": r["Ticker"],
            "Ürün": r["Ürün"],
            "Yahoo": ysym,
            "Durum": "OK" if note=="OK" else note,
            "Sektör": r.get("Sektör","-")
        })
    map_df = pd.DataFrame(rows)
    supported = map_df[map_df["Durum"]=="OK"]["Yahoo"].tolist()
    return map_df, supported

def fetch_yahoo_closes(symbols: list[str]) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(days=140)
    df = yf.download(symbols, start=start, end=end)["Close"].asfreq("B").ffill()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[1] for c in df.columns]
    return df.iloc[-50:]
