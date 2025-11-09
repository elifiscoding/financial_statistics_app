# 6 Haftalık Portföy Optimizasyonu (Streamlit)

Bu proje; hisse/ETF/FX/Futures listesi girmenizi, 50 iş günü verisi ile **momentum + volatilite** analizi yapmanızı, 
**Markowitz optimizasyonu** ile portföy ağırlıkları üretmenizi ve (hisse/ETF için) **opsiyon zinciri** üzerinden
6 haftaya hizalı **enstrüman önerileri** (hisse/ETF vs. opsiyon) sunmayı amaçlar. *Emir göndermez.*

## Hızlı Başlangıç
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Dosya Yapısı
- `streamlit_app.py` : Streamlit arayüz ve akış
- `app_modules/data.py` : Veri çekme & sembol eşleştirme
- `app_modules/optimize.py` : Getiri/kovaryans & optimizasyon
- `app_modules/options.py` : Opsiyon zinciri, BS grekleri, 6 haftalık öneri
- `app_modules/utils.py` : Yardımcılar (momentum, vol, formatlama)

## Notlar
- Opsiyonlar **yfinance** ile hisse/ETF sembollerinde çalışır. FX/Futures için zincir sınırlıdır; proxy ETF (örn. SPY, UUP) önerilir.
- Bu yazılım yalnızca analitik amaçlıdır; **yatırım tavsiyesi değildir**.
