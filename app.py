import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from data import generate_sales
from model import train_eval, forecast, detect_anomalies

st.set_page_config(page_title="Demand Forecasting & Anomaly Detection", page_icon="📈", layout="wide")


@st.cache_data
def load():
    return generate_sales()


@st.cache_resource(show_spinner="Training model...")
def fit(store):
    d = load().query("store == @store")[["date", "promo", "sales"]]
    return train_eval(d)


sales = load()
st.title("📈 Demand Forecasting & Anomaly Detection")
st.caption("Gradient-boosted time-series forecasting with lag features, a seasonal-naive baseline, "
           "what-if promotion scenarios and robust anomaly detection.")

with st.sidebar:
    st.header("Controls")
    store = st.selectbox("Store", sorted(sales["store"].unique()))
    horizon = st.slider("Forecast horizon (days)", 7, 60, 30)
    promo = st.checkbox("Simulate a promotion during forecast", value=False)
    z = st.slider("Anomaly sensitivity (z-score)", 2.5, 6.0, 3.5, 0.5)

d = sales.query("store == @store")[["date", "promo", "sales"]].reset_index(drop=True)
model, test, m = fit(store)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model MAPE", f"{m['model_mape']:.2f}%")
c2.metric("Baseline MAPE", f"{m['naive_mape']:.2f}%")
c3.metric("Model MAE", f"{m['model_mae']:.1f}")
c4.metric("MAE improvement vs baseline", f"{m['improvement_pct']:.1f}%")

tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "✅ Backtest", "🚨 Anomalies"])

with tab1:
    fc = forecast(model, d, horizon, int(promo))
    fig, ax = plt.subplots(figsize=(10, 4))
    recent = d.tail(90)
    ax.plot(recent["date"], recent["sales"], label="Actual")
    ax.plot(fc["date"], fc["forecast"], "--", label="Forecast")
    ax.legend(); ax.set_ylabel("Units sold")
    st.pyplot(fig)
    st.metric(f"Expected total demand ({horizon} days)", f"{fc['forecast'].sum():,.0f} units")
    st.download_button("Download forecast CSV", fc.to_csv(index=False), "forecast.csv")

with tab2:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test["date"], test["sales"], label="Actual")
    ax.plot(test["date"], test["pred"], label="Model")
    ax.plot(test["date"], test["naive"], ":", label="Seasonal naive")
    ax.legend()
    st.pyplot(fig)
    st.caption("Time-based split: the last 60 days are held out (no leakage).")

with tab3:
    a = detect_anomalies(d, z)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(a["date"], a["sales"], alpha=0.7)
    ax.scatter(a.loc[a.anomaly, "date"], a.loc[a.anomaly, "sales"], color="red", label="Anomaly")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(a.loc[a.anomaly, ["date", "sales", "score"]].round(2), use_container_width=True)
