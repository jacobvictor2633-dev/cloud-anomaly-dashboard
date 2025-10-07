"""
detect.py
Anomaly detection demo using IsolationForest. If scikit-learn is not available,
it falls back to a z-score method.
Usage:
    python detect.py --input sample_logs.csv --out anomalies.csv
Or import as a module.
"""
import argparse
import pandas as pd
import numpy as np

def detect_anomalies(df, features=None, contamination=0.01):
    # Try sklearn's IsolationForest if available, else fallback to z-score
    if features is None:
        features = ["CPUUtilization", "MemoryUtilization", "NetworkIn", "NetworkOut", "DiskIO", "Cost"]
    X = df[features].values
    try:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        preds = model.fit_predict(X)
        df["anomaly_iforest"] = (preds == -1).astype(int)
    except Exception as e:
        # fallback: simple z-score on each feature (mark row anomalous if any feature > 4 sigma)
        z = np.abs((X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-9))
        df["anomaly_zscore"] = (z > 4).any(axis=1).astype(int)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_logs.csv")
    parser.add_argument("--out", default="anomalies.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.input, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp")
    out = detect_anomalies(df)
    out.to_csv(args.out, index=False)
    print(f"Saved anomalies to {args.out}")
