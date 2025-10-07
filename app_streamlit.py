"""
app_streamlit.py
A simple Streamlit app skeleton to upload a CSV log file and run the pipeline.
Run with: streamlit run app_streamlit.py
"""
import streamlit as st
import pandas as pd
import io
import subprocess
import sys

st.title("Cloud Log Uploader — Insights Demo")

uploaded = st.file_uploader("Upload CSV log file (sample format provided)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["Timestamp"])
    st.write("Preview of uploaded logs:")
    st.dataframe(df.head(50))
    st.write("Running preprocessing...")
    # Use local copy of helper scripts if available, else run inline
    st.write("Running anomaly detection (IsolationForest if available)...")
    try:
        from sklearn.ensemble import IsolationForest
        features = ["CPUUtilization", "MemoryUtilization", "NetworkIn", "NetworkOut", "DiskIO", "Cost"]
        X = df[features].fillna(0).values
        model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
        preds = model.fit_predict(X)
        df["anomaly"] = (preds == -1).astype(int)
    except Exception as e:
        st.warning("sklearn not available — falling back to z-score method")
        import numpy as np
        features = ["CPUUtilization", "MemoryUtilization", "NetworkIn", "NetworkOut", "DiskIO", "Cost"]
        X = df[features].fillna(0).values
        z = np.abs((X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-9))
        df["anomaly"] = (z > 4).any(axis=1).astype(int)
    st.write("Anomalies found:")
    st.dataframe(df[df["anomaly"]==1].head(200))
    st.write("Generating simple cost forecast (7-day mean)...")
    df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
    daily = df.groupby(["ResourceId", "date"])["Cost"].sum().reset_index()
    forecasts = []
    for rid, g in daily.groupby("ResourceId"):
        last_mean = g["Cost"].tail(7).mean() if len(g) >= 1 else 0.0
        forecasts.append({"ResourceId": rid, "predicted_next_7day_daily_cost": last_mean})
    st.table(pd.DataFrame(forecasts))
    st.success("Done — download anomalies or recommendations using the exported CSV buttons (not implemented in this skeleton).")
