# 📈 Demand Forecasting & Anomaly Detection

End-to-end time-series ML project: feature engineering, gradient-boosted forecasting, leakage-free backtesting
against a seasonal-naive baseline, what-if promotion scenarios, robust anomaly detection, tests, CI and Docker.

**Live demo:** _add your Streamlit link here_

## Results (60-day time-based holdout, synthetic retail data)
| Store | Model MAPE | Baseline MAPE | MAE improvement |
|-------|-----------|---------------|-----------------|
| A | 5.0% | 7.4% | 33% |
| B | 4.5% | 7.4% | 40% |
| C | 6.0% | 10.2% | 39% |

> Data is synthetic (trend + weekly/yearly seasonality + promotions + noise + injected anomalies) so the project runs anywhere with no downloads.
> `generate_sales()` can be swapped for a real CSV with columns `date, store, promo, sales`.

## Approach
- **Features:** lags (1/7/14/28), rolling means, calendar features, promo flag
- **Model:** `HistGradientBoostingRegressor`; recursive multi-step forecasting
- **Validation:** time-based split (no leakage), compared with seasonal-naive baseline
- **Anomalies:** robust z-score (MAD) on log-ratio vs same-weekday median of previous 4 weeks
- **Engineering:** pytest tests, GitHub Actions CI, Dockerfile

## Run
```bash
pip install -r requirements-dev.txt
pytest -q
streamlit run app.py
```
Docker: `docker build -t forecast . && docker run -p 8501:8501 forecast`

## Structure
`app.py` dashboard · `model.py` features/model/forecast/anomalies · `data.py` data generator · `tests/` · `.github/workflows/ci.yml` · `Dockerfile`
