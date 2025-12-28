#!/usr/bin/env python3
"""colab_workflow_v13.py

V13 "The Signal Miner" (Unsupervised Anomaly Detection)

Objective:
- Stop trying to "teach" the model what a reversal is (supervised learning).
- Instead, let the model "discover" rare market states (unsupervised anomaly detection).
- Hypothesis: Reversals are mathematically "anomalous" events where volume, volatility, and momentum diverge from the norm.

Method:
- Algorithm: Isolation Forest (effective for high-dimensional anomaly detection).
- Features: RSI Divergence, Volume Z-Score, Volatility spread, Momentum acceleration.
- Output: Anomaly Score (-1 = Anomaly/Reversal, 1 = Normal/Trend).

Run on Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v13.py | python3 - \
  --symbol BTCUSDT --interval 15m --contamination 0.05

Artifacts:
- plot : ./all_models/models_v13/{symbol}/plots/anomaly_signals.png
- csv  : ./all_models/models_v13/{symbol}/anomalies.csv (Rows where anomaly detected)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------------
# Advanced Feature Engineering
# ------------------------------
def add_features_miner(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    
    # Parse time
    time_col = next((c for c in ["open_time", "opentime", "timestamp", "time", "date"] if c in d.columns), None)
    if not time_col: raise ValueError("Missing time column")
    if pd.api.types.is_numeric_dtype(d[time_col]):
        ts = d[time_col].astype("int64")
        unit = "ms" if ts.median() > 10**12 else "s"
        d["open_time"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        d["open_time"] = pd.to_datetime(d[time_col], utc=True, errors="coerce")

    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    
    d = d.dropna().sort_values("open_time").reset_index(drop=True)

    # 1. Volatility Anomalies
    d["range"] = (d["high"] - d["low"]) / d["close"]
    d["range_z"] = (d["range"] - d["range"].rolling(50).mean()) / d["range"].rolling(50).std()
    
    # 2. Volume Anomalies
    d["vol_z"] = (d["volume"] - d["volume"].rolling(50).mean()) / d["volume"].rolling(50).std()
    
    # 3. Momentum Acceleration (Second derivative of price)
    # roc1 = velocity, roc_diff = acceleration
    roc = d["close"].pct_change(5)
    d["accel"] = roc.diff()
    
    # 4. RSI Extremes
    delta = d["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    d["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    # Distance from neutral 50 (abs value)
    d["rsi_dist"] = (d["rsi"] - 50).abs()

    d = d.dropna().reset_index(drop=True)
    
    # We only use features that describe "state intensity", not direction
    feature_cols = ["range_z", "vol_z", "accel", "rsi_dist"]
    return d, feature_cols

# ------------------------------
# Main
# ------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    p.add_argument("--contamination", type=float, default=0.05) # Top 5% rarest events
    args = p.parse_args()

    print("\n[1/3] Data Load & Feature Eng")
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="zongowo111/cpb-models", repo_type="dataset", allow_patterns=None, ignore_patterns=None)
    
    csv_file = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv") and args.symbol in f and args.interval in f:
                csv_file = os.path.join(root, f)
                break
        if csv_file: break
    if not csv_file: raise ValueError("No CSV found")
    
    df = pd.read_csv(csv_file)
    df, features = add_features_miner(df)
    
    # Scale features
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"[2/3] Mining Anomalies (Isolation Forest, contamination={args.contamination})")
    # Isolation Forest detects data points that are "few and different"
    iso = IsolationForest(contamination=args.contamination, random_state=42, n_jobs=-1)
    df["anomaly"] = iso.fit_predict(X_scaled) # -1 = Anomaly, 1 = Normal
    df["score"] = iso.decision_function(X_scaled) # Lower score = more anomalous

    # Filter only anomalies
    anomalies = df[df["anomaly"] == -1].copy()
    print(f"Found {len(anomalies)} anomalies out of {len(df)} candles.")

    print("[3/3] Visualize & Save")
    out_dir = f"./all_models/models_v13/{args.symbol}"
    _safe_mkdir(os.path.join(out_dir, "plots"))
    
    # Save CSV of signals
    anomalies.to_csv(os.path.join(out_dir, "anomalies.csv"), index=False)
    
    # Plot last 1000 candles to see if anomalies align with reversals
    subset = df.iloc[-1000:].reset_index(drop=True)
    subset_anom = subset[subset["anomaly"] == -1]
    
    plt.figure(figsize=(14, 7))
    plt.plot(subset.index, subset["close"], color="gray", alpha=0.6, label="Price")
    
    # Color anomalies by whether they were high or low relative to recent price
    # Simple heuristic for coloring: if anomaly candle close < SMA(20) -> Potential Buy (Panic Dump)
    # If anomaly candle close > SMA(20) -> Potential Sell (Blow-off Top)
    sma20 = subset["close"].rolling(20).mean()
    
    for idx in subset_anom.index:
        price = subset.loc[idx, "close"]
        ma = sma20.loc[idx] if not pd.isna(sma20.loc[idx]) else price
        
        if price < ma:
            color = "green" # Panic Low
            marker = "^"
        else:
            color = "red"   # Blow-off High
            marker = "v"
            
        plt.scatter(idx, price, color=color, marker=marker, s=50, zorder=5)
        
    plt.title(f"V13 Signal Miner: Unsupervised Anomaly Detection (Contamination={args.contamination})")
    plt.legend(["Price", "Anomalous Event (Potential Reversal)"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", "anomaly_signals.png"))
    print(f"Saved plot to {out_dir}/plots/anomaly_signals.png")

if __name__ == "__main__":
    main()
