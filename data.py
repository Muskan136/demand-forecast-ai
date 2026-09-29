import numpy as np
import pandas as pd


def generate_sales(start="2023-01-01", days=900, seed=42):
    """Synthetic retail sales: trend + weekly/yearly seasonality + promos + noise + injected anomalies."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    frames = []
    for store, base in {"Store A": 200, "Store B": 150, "Store C": 100}.items():
        trend = base * (1 + 0.0004 * t)
        weekly = 1 + 0.25 * np.isin(dates.dayofweek, [4, 5, 6])
        yearly = 1 + 0.15 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        promo = (rng.random(days) < 0.08).astype(int)
        sales = np.asarray(trend * weekly * yearly * (1 + 0.3 * promo) * rng.normal(1, 0.05, days), dtype=float)
        idx = rng.choice(np.arange(60, days - 5), 6, replace=False)
        sales[idx] *= rng.choice([0.4, 1.8], 6)  # injected anomalies
        frames.append(pd.DataFrame({"date": dates, "store": store, "promo": promo, "sales": sales.round(1)}))
    return pd.concat(frames, ignore_index=True)
