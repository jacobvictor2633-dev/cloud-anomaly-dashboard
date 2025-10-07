"""
preprocess.py
Simple preprocessing utilities for cloud log CSVs.
Usage (as module):
    from preprocess import load_and_preprocess
    df = load_and_preprocess("sample_logs.csv")
"""
import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    # Basic cleaning
    numeric_cols = ["CPUUtilization", "MemoryUtilization", "NetworkIn", "NetworkOut", "DiskIO", "Cost"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # Add date and hour features
    df["date"] = df["Timestamp"].dt.date
    df["hour"] = df["Timestamp"].dt.hour
    # compute rolling features per resource (1h and 24h mean where meaningful)
    df = df.set_index("Timestamp")
    df = df.groupby("ResourceId").apply(lambda g: g.assign(
        cpu_roll_3h = g["CPUUtilization"].rolling("3h").mean().fillna(g["CPUUtilization"]),
        cpu_roll_24h = g["CPUUtilization"].rolling("24h").mean().fillna(g["CPUUtilization"])
    )).reset_index(level=0, drop=False)
    return df
