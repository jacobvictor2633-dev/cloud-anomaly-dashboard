"""
recommend.py
Generate simple human-readable recommendations based on recent usage.
Usage:
    python recommend.py --input sample_logs.csv --out recommendations.csv
"""
import argparse
import pandas as pd

def generate_recommendations(df):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")
    recs = []
    for rid, g in df.groupby("ResourceId"):
        last_7d = g[g["Timestamp"] >= (g["Timestamp"].max() - pd.Timedelta(days=7))]
        avg_cpu = last_7d["CPUUtilization"].mean() if len(last_7d)>0 else 0.0
        idle_hours = (last_7d["CPUUtilization"] < 1.0).sum()
        avg_cost = last_7d["Cost"].mean() if len(last_7d)>0 else 0.0
        if avg_cpu < 5.0:
            recs.append({"ResourceId": rid, "recommendation": f"Average CPU {avg_cpu:.2f}% → consider downsizing or stopping idle resource.", "estimated_monthly_savings_USD": round(avg_cost*24*30, 2)})
        if idle_hours >= 48:
            recs.append({"ResourceId": rid, "recommendation": f"Resource has {idle_hours} idle hours in last 7 days → consider termination or scheduled shutdown.", "estimated_monthly_savings_USD": round(avg_cost*24*30, 2)})
        # Simple suggestion about reserved/spot
        if avg_cost > 0.05:
            recs.append({"ResourceId": rid, "recommendation": f"Consider reserved or spot instances for long-running workloads to save costs.", "estimated_monthly_savings_USD": round(0.1*avg_cost*24*30, 2)})
    return pd.DataFrame(recs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_logs.csv")
    parser.add_argument("--out", default="recommendations.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    recs = generate_recommendations(df)
    recs.to_csv(args.out, index=False)
    print(f"Saved recommendations to {args.out}")
