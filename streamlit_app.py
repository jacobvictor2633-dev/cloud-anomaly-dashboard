# streamlit_app.py
# AI-driven Cloud Cost Optimization & Anomaly Detection Dashboard

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import timedelta

st.set_page_config(
    page_title="Cloud Cost Anomaly Detection",
    layout="wide",
    page_icon="☁️"
)

st.title("☁️ AI-Driven Cloud Cost Optimization & Anomaly Detection Dashboard")
st.markdown("Upload your **multi-cloud log file (CSV)** to get instant insights on cost and anomalies.")

# ------------------------------
# 1️⃣ File Upload
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Timestamp"])
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df.head(20), use_container_width=True)

    # ------------------------------
    # 2️⃣ Data Cleaning
    # ------------------------------
    numeric_cols = ["CPUUtilization", "MemoryUtilization", "NetworkIn", "NetworkOut", "DiskIO", "Cost"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ------------------------------
    # 3️⃣ Anomaly Detection (IsolationForest)
    # ------------------------------
    st.subheader("🚨 Anomaly Detection")
    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    df["anomaly"] = model.fit_predict(df[numeric_cols])
    df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)
    anomalies = df[df["anomaly"] == 1]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("🔍 Total Records", len(df))
    with c2:
        st.metric("⚠️ Detected Anomalies", len(anomalies))

    st.dataframe(anomalies.head(20), use_container_width=True)

    # ------------------------------
    # 4️⃣ Cost Forecasting (Simple Moving Average)
    # ------------------------------
    st.subheader("💰 7-Day Cost Forecast")
    df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
    daily_cost = df.groupby(["ResourceId", "date"])["Cost"].sum().reset_index()

    forecast_list = []
    for rid, g in daily_cost.groupby("ResourceId"):
        g = g.sort_values("date")
        mean_last7 = g["Cost"].tail(7).mean()
        next7 = [g["date"].max() + timedelta(days=i) for i in range(1, 8)]
        for d in next7:
            forecast_list.append({"ResourceId": rid, "PredictedDate": d, "PredictedCost": mean_last7})
    forecast_df = pd.DataFrame(forecast_list)

    st.dataframe(forecast_df.head(20), use_container_width=True)

    # ------------------------------
    # 5️⃣ Recommendations
    # ------------------------------
    st.subheader("🧠 Optimization Recommendations")

    recs = []
    for rid, g in df.groupby("ResourceId"):
        recent = g[g["Timestamp"] >= (g["Timestamp"].max() - timedelta(days=7))]
        avg_cpu = recent["CPUUtilization"].mean()
        avg_cost = recent["Cost"].mean()
        idle_hours = (recent["CPUUtilization"] < 1.0).sum()

        if avg_cpu < 5:
            recs.append({
                "ResourceId": rid,
                "Recommendation": f"Low CPU usage ({avg_cpu:.1f}%) — consider downsizing or auto-stopping.",
                "Potential Monthly Savings ($)": round(avg_cost * 24 * 30, 2)
            })
        if idle_hours > 48:
            recs.append({
                "ResourceId": rid,
                "Recommendation": f"Idle for {idle_hours} hours — enable scheduled shutdown or termination.",
                "Potential Monthly Savings ($)": round(avg_cost * 24 * 30, 2)
            })
        if avg_cost > 0.05:
            recs.append({
                "ResourceId": rid,
                "Recommendation": "Use reserved/spot instances for long-running workloads.",
                "Potential Monthly Savings ($)": round(avg_cost * 24 * 30 * 0.1, 2)
            })

    rec_df = pd.DataFrame(recs)
    if len(rec_df) > 0:
        st.dataframe(rec_df, use_container_width=True)
    else:
        st.info("No specific recommendations — usage seems balanced ✅")

    # ------------------------------
    # 6️⃣ Visualization
    # ------------------------------
    st.subheader("📈 Cost Trend and Anomalies Visualization")

    import altair as alt
    chart = alt.Chart(df).mark_line().encode(
        x='Timestamp:T',
        y='Cost:Q',
        color='ResourceId:N'
    ) + alt.Chart(df[df["anomaly"] == 1]).mark_circle(size=60, color='red').encode(
        x='Timestamp:T',
        y='Cost:Q',
        tooltip=['ResourceId', 'Cost', 'CPUUtilization', 'MemoryUtilization']
    )
    st.altair_chart(chart.interactive(), use_container_width=True)

    st.success("✅ Analysis complete! Scroll up to explore results.")
else:
    st.info("👆 Upload a log file (CSV) to begin analysis.")
