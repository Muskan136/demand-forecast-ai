import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["lag_1", "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28",
            "dow", "month", "dayofyear", "promo"]


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    """d: single-store frame with columns date, promo, sales (sorted by date)."""
    d = d.sort_values("date").reset_index(drop=True).copy()
    s = d["sales"]
    for lag in (1, 7, 14, 28):
        d[f"lag_{lag}"] = s.shift(lag)
    d["roll_mean_7"] = s.shift(1).rolling(7).mean()
    d["roll_mean_28"] = s.shift(1).rolling(28).mean()
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["dayofyear"] = d["date"].dt.dayofyear
    return d


def train_eval(d: pd.DataFrame, test_days: int = 60):
    """Time-based split. Returns model, test frame with predictions, metrics vs seasonal-naive baseline."""
    f = make_features(d).dropna(subset=FEATURES)
    train, test = f.iloc[:-test_days], f.iloc[-test_days:].copy()
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
    model.fit(train[FEATURES], train["sales"])
    test["pred"] = model.predict(test[FEATURES])
    test["naive"] = test["lag_7"]  # seasonal naive baseline: same weekday last week
    y = test["sales"]

    def mape(p):
        return float(np.mean(np.abs((y - p) / y)) * 100)

    metrics = {
        "model_mae": float(np.mean(np.abs(y - test["pred"]))),
        "naive_mae": float(np.mean(np.abs(y - test["naive"]))),
        "model_mape": mape(test["pred"]),
        "naive_mape": mape(test["naive"]),
    }
    metrics["improvement_pct"] = 100 * (1 - metrics["model_mae"] / metrics["naive_mae"])
    return model, test, metrics


def forecast(model, d: pd.DataFrame, horizon: int = 30, promo: int = 0) -> pd.DataFrame:
    """Recursive multi-step forecast; promo=1 simulates a promotion running throughout the horizon."""
    hist = d[["date", "promo", "sales"]].copy()
    for _ in range(horizon):
        nxt = hist["date"].max() + pd.Timedelta(days=1)
        hist = pd.concat([hist, pd.DataFrame({"date": [nxt], "promo": [promo], "sales": [np.nan]})],
                         ignore_index=True)
        row = make_features(hist).iloc[[-1]]
        hist.loc[hist.index[-1], "sales"] = float(model.predict(row[FEATURES])[0])
    return hist.tail(horizon).rename(columns={"sales": "forecast"}).reset_index(drop=True)


def detect_anomalies(d: pd.DataFrame, z: float = 3.5) -> pd.DataFrame:
    """Robust z-score of log(sales / median of same weekday over previous 4 weeks). Promo days excluded."""
    d = d.sort_values("date").reset_index(drop=True).copy()
    s = d["sales"]
    expected = pd.concat([s.shift(k) for k in (7, 14, 21, 28)], axis=1).median(axis=1)
    ratio = np.log(s / expected)
    med = ratio.median()
    mad = (ratio - med).abs().median() * 1.4826
    d["score"] = (ratio - med) / mad
    d["anomaly"] = (d["score"].abs() > z) & (d["promo"] == 0)
    return d
