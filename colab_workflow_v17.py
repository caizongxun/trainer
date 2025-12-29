#!/usr/bin/env python3
"""colab_workflow_v17.py

V17 "The Grand Unification" (Regime-Aware Rule Discovery)

Objective:
- Answer the user's ultimate request: "First classify the regime, then find formulas for EACH regime separately."
- Step 1: Use V15 logic (GMM) to segment the 10,000 candles into "Trend" and "Range".
- Step 2: Split the dataset into two: `df_trend` and `df_range`.
- Step 3: Use V14 logic (Decision Tree) to discover reversal rules SPECIFIC to each dataset.
    - Find "Range Reversal Formula" (likely BB mean reversion).
    - Find "Trend Reversal Formula" (likely divergence or climax).

Run on Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v17.py | python3 - \
  --symbol BTCUSDT --interval 15m

Artifacts:
- text : ./all_models/models_v17/{symbol}/final_formulas.txt
- plot : ./all_models/models_v17/{symbol}/plots/regime_split_viz.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.preprocessing import StandardScaler

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------------
# Feature Engineering (Combined V14 + V15)
# ------------------------------
def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

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

def add_features_grand(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    
    # Time
    time_col = next((c for c in ["open_time", "opentime", "timestamp", "time", "date"] if c in d.columns), None)
    if pd.api.types.is_numeric_dtype(d[time_col]):
        ts = d[time_col].astype("int64")
        unit = "ms" if ts.median() > 10**12 else "s"
        d["open_time"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        d["open_time"] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    
    d = d.dropna().sort_values("open_time").reset_index(drop=True)

    # --- Regime Features (for GMM) ---
    d["adx"] = _adx(d)
    
    # Volatility Z
    d["range"] = (d["high"] - d["low"]) / d["close"]
    d["range_z"] = (d["range"] - d["range"].rolling(50).mean()) / d["range"].rolling(50).std()
    
    # --- Rule Features (for Tree) ---
    d["rsi"] = _wilder_rsi(d["close"])
    d["rsi_slope"] = d["rsi"].diff(3)
    
    d["vol_z"] = (d["volume"] - d["volume"].rolling(50).mean()) / (d["volume"].rolling(50).std() + 1e-12)
    
    bb_mean = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_pct"] = (d["close"] - (bb_mean - 2*bb_std)) / (4*bb_std + 1e-12)
    d["bb_width"] = (4 * bb_std) / bb_mean
    
    d["roc_6"] = d["close"].pct_change(6) * 100
    d["upper_shadow"] = (d["high"] - np.maximum(d["close"], d["open"])) / d["close"]
    d["lower_shadow"] = (np.minimum(d["close"], d["open"]) - d["low"]) / d["close"]

    d = d.dropna().reset_index(drop=True)
    
    regime_cols = ["adx", "range_z"]
    rule_cols = ["rsi", "rsi_slope", "vol_z", "bb_pct", "bb_width", "roc_6", "upper_shadow", "lower_shadow"]
    
    return d, regime_cols, rule_cols

# ------------------------------
# Labeling
# ------------------------------
def label_pivots_strict(df: pd.DataFrame, lookback=12, lookahead=12):
    highs = df["high"].values
    lows = df["low"].values
    labels = np.zeros(len(df), dtype=int)
    for i in range(lookback, len(df) - lookahead):
        if highs[i] == np.max(highs[i-lookback : i+lookahead+1]):
            labels[i] = 2 # Sell
        elif lows[i] == np.min(lows[i-lookback : i+lookahead+1]):
            labels[i] = 1 # Buy
    return labels

# ------------------------------
# Rule Extraction Helper
# ------------------------------
def extract_rules(X, y, feature_names, class_name, depth=3):
    # Only if we have enough samples
    if len(y) < 50 or np.sum(y) < 10:
        return [f"Not enough data to find {class_name} rules."]
        
    clf = DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=42)
    clf.fit(X, y)
    
    tree_ = clf.tree_
    rules = []
    
    def recurse(node, path_str):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            thresh = tree_.threshold[node]
            recurse(tree_.children_left[node], path_str + [f"{name} <= {thresh:.2f}"])
            recurse(tree_.children_right[node], path_str + [f"{name} > {thresh:.2f}"])
        else:
            # Leaf
            counts = tree_.value[node][0] # [Neg, Pos]
            total = counts[0] + counts[1]
            prob = counts[1] / total
            if prob > 0.60: # Confidence threshold
                rules.append(f"IF {' AND '.join(path_str)} THEN {class_name} (Prob: {prob:.2f}, Samples: {int(counts[1])})")
    
    recurse(0, [])
    return rules

# ------------------------------
# Main
# ------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    args = p.parse_args()

    print("\n[1/4] Data & Features")
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
    df, regime_cols, rule_cols = add_features_grand(df)
    
    # Label Pivots (Truth)
    df["label"] = label_pivots_strict(df, 24, 24)
    
    print("\n[2/4] Regime Classification (GMM)")
    # We want 2 regimes: 0=Range, 1=Trend
    # We assume 'Range' has lower ADX/Range_Z
    X_regime = df[regime_cols].values
    scaler = StandardScaler()
    X_regime_scaled = scaler.fit_transform(X_regime)
    
    gmm = GaussianMixture(n_components=2, random_state=42)
    df["regime_raw"] = gmm.fit_predict(X_regime_scaled)
    
    # Check which label is 'Trend' (Higher ADX)
    mean_adx_0 = df[df["regime_raw"] == 0]["adx"].mean()
    mean_adx_1 = df[df["regime_raw"] == 1]["adx"].mean()
    
    if mean_adx_1 > mean_adx_0:
        df["regime"] = df["regime_raw"] # 1 is Trend
    else:
        df["regime"] = 1 - df["regime_raw"] # Flip so 1 is Trend
        
    print(f"Regime 0 (Range) avg ADX: {df[df['regime']==0]['adx'].mean():.2f}")
    print(f"Regime 1 (Trend) avg ADX: {df[df['regime']==1]['adx'].mean():.2f}")
    
    # Split Data
    df_range = df[df["regime"] == 0].copy()
    df_trend = df[df["regime"] == 1].copy()
    
    print(f"Range Samples: {len(df_range)}, Trend Samples: {len(df_trend)}")
    
    print("\n[3/4] Discovering Formulas for EACH Regime")
    
    out_dir = f"./all_models/models_v17/{args.symbol}"
    _safe_mkdir(os.path.join(out_dir, "plots"))
    report_path = os.path.join(out_dir, "final_formulas.txt")
    
    with open(report_path, "w") as f:
        f.write(f"=== THE GRAND UNIFICATION FORMULAS ({args.symbol}) ===\n")
        f.write("Method: Unsupervised Regime Split (GMM) -> Supervised Rule Discovery (Tree)\n\n")
        
        # --- Range Rules ---
        f.write("--- REGIME: RANGE (Low Volatility/ADX) ---\n")
        f.write("Strategy: Look for Mean Reversion (Buying Lows, Selling Highs)\n\n")
        
        # Range Buy
        range_buy_rules = extract_rules(df_range[rule_cols], (df_range["label"]==1).astype(int), rule_cols, "BUY")
        for r in range_buy_rules: f.write(r + "\n")
        f.write("\n")
        
        # Range Sell
        range_sell_rules = extract_rules(df_range[rule_cols], (df_range["label"]==2).astype(int), rule_cols, "SELL")
        for r in range_sell_rules: f.write(r + "\n")
        f.write("\n")
        
        # --- Trend Rules ---
        f.write("--- REGIME: TREND (High Volatility/ADX) ---\n")
        f.write("Strategy: Look for Trend Reversals/Exhaustion (Harder to catch!)\n\n")
        
        # Trend Buy (catching the bottom of a crash)
        trend_buy_rules = extract_rules(df_trend[rule_cols], (df_trend["label"]==1).astype(int), rule_cols, "BUY")
        for r in trend_buy_rules: f.write(r + "\n")
        f.write("\n")
        
        # Trend Sell (catching the top of a pump)
        trend_sell_rules = extract_rules(df_trend[rule_cols], (df_trend["label"]==2).astype(int), rule_cols, "SELL")
        for r in trend_sell_rules: f.write(r + "\n")
        f.write("\n")

    print("\n[4/4] Visualization")
    # Visualize where the regimes split
    subset = df.iloc[-1000:]
    plt.figure(figsize=(14, 6))
    plt.plot(subset.index, subset["close"], color="gray", alpha=0.5)
    
    # Highlight Trend Regime
    trend_indices = subset[subset["regime"] == 1].index
    if len(trend_indices) > 0:
        plt.scatter(trend_indices, subset.loc[trend_indices, "close"], color="purple", s=1, alpha=0.3, label="Trend Regime")
        
    plt.title("Regime Classification: Purple areas = Trend (Use Trend Formula), Gray = Range (Use Range Formula)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "plots", "regime_split_viz.png"))
    
    print(f"Discovery Complete. Check {report_path}")

if __name__ == "__main__":
    main()
