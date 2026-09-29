import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data import generate_sales
from model import train_eval, forecast, detect_anomalies

D = generate_sales().query("store == 'Store A'")[["date", "promo", "sales"]].reset_index(drop=True)


def test_model_beats_baseline():
    _, _, m = train_eval(D)
    assert m["model_mae"] < m["naive_mae"]


def test_forecast_length_and_positive():
    model, _, _ = train_eval(D)
    fc = forecast(model, D, horizon=14)
    assert len(fc) == 14 and (fc["forecast"] > 0).all()


def test_anomalies_found():
    a = detect_anomalies(D)
    assert 3 <= a["anomaly"].sum() <= 15
