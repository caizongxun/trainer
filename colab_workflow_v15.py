#!/usr/bin/env python3
"""colab_workflow_v15.py

V15 "The Regime Filter" (Unsupervised Regime Classification)

Objective:
- Solve the "False Signals in Trend" problem.
- Use Unsupervised Clustering (GMM/K-Means) to classify market states into:
  0: Chop / Noise (Ignore mean-reversion signals here? Or maybe use tighter stops)
  1: Bull Trend (Only take Longs)
  2: Bear Trend (Only take Shorts)
  3: Volatile Reversal (High risk/reward)

Method:
- Features: ADX, Choppiness Index, Volatility Z-Score, SMA Slope.
- Algorithm: Gaussian Mixture Model (GMM) to find 4 hidden states.
- Output: A "Regime Map" for the 10,000 candles.

Run on Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v15.py | python3 - \
  --symbol BTCUSDT --interval 15m

Artifacts:
- plot : ./all_models/models_v15/{symbol}/plots/regime_map.png
- csv  : ./all_models/models_v15/{symbol}/regime_labels.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------------
# Feature Engineering (Regime Focus)
# ------------------------------
def _adx(df, period=14):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx.fillna(0)

def _choppiness(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(period).sum()
    high_max = df['high'].rolling(period).max()
    low_min = df['low'].rolling(period).min()
    
    chop = 100 * np.log10(atr_sum / (high_max - low_min + 1e-12)) / np.log10(period)
    return chop.fillna(50)

def add_features_regime(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    
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

    # 1. Trend Strength
    d["adx"] = _adx(d)
    d["chop"] = _choppiness(d)
    
    # 2. Volatility
    d["range"] = (d["high"] - d["low"]) / d["close"]
    d["range_z"] = (d["range"] - d["range"].rolling(50).mean()) / d["range"].rolling(50).std()
    
    # 3. Directional Bias (SMA Slope)
    sma50 = d["close"].rolling(50).mean()
    d["trend_bias"] = (d["close"] - sma50) / sma50

    d = d.dropna().reset_index(drop=True)
    
    feature_cols = ["adx", "chop", "range_z", "trend_bias"]
    return d, feature_cols

# ------------------------------
# Main
# ------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    p.add_argument("--n_components", type=int, default=4) # 4 regimes
    args = p.parse_args()

    print("\n[1/3] Data & Feature Eng")
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
    df, features = add_features_regime(df)
    
    # Clustering
    print(f"[2/3] Clustering Regimes (GMM, n={args.n_components})")
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    gmm = GaussianMixture(n_components=args.n_components, random_state=42)
    df["regime"] = gmm.fit_predict(X_scaled)
    
    # Interpret Regimes (Heuristic)
    # We want to know which cluster is "Trend" and which is "Chop"
    # Calculate mean ADX for each cluster
    cluster_stats = df.groupby("regime")[["adx", "chop", "trend_bias"]].mean()
    print("\nRegime Stats:")
    print(cluster_stats)
    
    # Sort labels so that 0 is lowest ADX (Chop) and N is highest ADX (Trend)
    # This makes plotting consistent
    sorted_indices = cluster_stats.sort_values("adx").index
    mapping = {old: new for new, old in enumerate(sorted_indices)}
    df["regime_sorted"] = df["regime"].map(mapping)
    
    print("\n[3/3] Visualize & Save")
    out_dir = f"./all_models/models_v15/{args.symbol}"
    _safe_mkdir(os.path.join(out_dir, "plots"))
    
    df.to_csv(os.path.join(out_dir, "regime_labels.csv"), index=False)
    
    # Plot last 1000 candles
    subset = df.iloc[-1000:].reset_index(drop=True)
    price = subset["close"].values
    regime = subset["regime_sorted"].values # 0=Weakest Trend, 3=Strongest Trend
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Color price by regime
    # 0 (Chop) = Gray
    # 1 (Weak Trend) = Yellow
    # 2 (Strong Trend) = Orange
    # 3 (Extreme Trend) = Red
    colors = ['lightgray', 'gold', 'orange', 'red']
    
    for i in range(len(subset) - 1):
        r = regime[i]
        c = colors[r] if r < len(colors) else 'black'
        ax1.plot(subset.index[i:i+2], price[i:i+2], color=c, linewidth=1.5)
        
    ax1.set_title(f"V15 Regime Filter: Gray=Chop, Red=Strong Trend ({args.symbol})")
    ax1.set_ylabel("Price")
    
    # Plot ADX and Chop below for reference
    ax2.plot(subset["adx"], label="ADX (Trend Strength)", color="blue", alpha=0.6)
    ax2.plot(subset["chop"], label="Chop Index (Volatility)", color="green", alpha=0.6)
    ax2.axhline(25, linestyle="--", color="gray", alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", "regime_map.png"))
    print(f"Saved plot to {out_dir}/plots/regime_map.png")

if __name__ == "__main__":
    main()
