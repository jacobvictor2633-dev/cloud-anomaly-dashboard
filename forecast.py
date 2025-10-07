"""
forecast.py
Simple cost forecasting using rolling-window mean per resource (no external libs required).
Usage:
    python forecast.py --input sample_logs.csv --out forecast.csv --horizon 7
"""
import argparse
import pandas as pd

def forecast_costs(df, horizon_days=7):
    # Aggregate daily cost per resource and forecast the next horizon_days using last 7-day mean
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["date"] = df["Timestamp"].dt.date
    daily = df.groupby(["ResourceId", "date"])["Cost"].sum().reset_index()
    forecasts = []
    for rid, g in daily.groupby("ResourceId"):
        g = g.sort_values("date")
        last_mean = g["Cost"].tail(7).mean() if len(g) >= 1 else 0.0
        for i in range(1, horizon_days+1):
            forecasts.append({"ResourceId": rid, "day_offset": i, "predicted_cost": last_mean})
    return pd.DataFrame(forecasts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_logs.csv")
    parser.add_argument("--out", default="forecast.csv")
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    forecast_df = forecast_costs(df, horizon_days=args.horizon)
    forecast_df.to_csv(args.out, index=False)
    print(f"Saved forecast to {args.out}")
