#!/usr/bin/env python3
"""colab_workflow_v14.py

V14 "The Rule Discovery Engine"

Objective:
- Instead of a "Black Box" Neural Network, use "White Box" Explainable AI (Decision Trees & XGBoost) to discover rules.
- Output human-readable rules like: "IF RSI < 25 AND Volume_Z > 2.5 THEN Buy".
- This answers the user's request to "find a formula" or "discover laws" from the 10,000 candles.

Method:
1. Label "True Reversals" strictly (ZigZag with lookahead).
2. Train a Decision Tree (max_depth=3) to visualize the simplest "If-Then" logic that works.
3. Train XGBoost to rank features by importance.
4. Export the rules as text.

Run on Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v14.py | python3 - \
  --symbol BTCUSDT --interval 15m

Artifacts:
- text : ./all_models/models_v14/{symbol}/discovered_rules.txt
- plot : ./all_models/models_v14/{symbol}/plots/tree_viz.png
- plot : ./all_models/models_v14/{symbol}/plots/feature_importance.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import StandardScaler

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------------
# Feature Engineering (Focus on "Causal" Indicators)
# ------------------------------
def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def add_features_discovery(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    
    # Time parsing
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

    # 1. RSI & RSI Slope
    d["rsi"] = _wilder_rsi(d["close"])
    d["rsi_slope"] = d["rsi"].diff(3)

    # 2. Volume Anomaly (Z-score)
    d["vol_z"] = (d["volume"] - d["volume"].rolling(50).mean()) / (d["volume"].rolling(50).std() + 1e-12)

    # 3. Bollinger %B (Position within bands)
    bb_mean = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_pct"] = (d["close"] - (bb_mean - 2*bb_std)) / (4*bb_std + 1e-12) # 0=Lower, 1=Upper, <0=Oversold, >1=Overbought
    d["bb_width"] = (4 * bb_std) / bb_mean

    # 4. Momentum (ROC)
    d["roc_6"] = d["close"].pct_change(6) * 100
    
    # 5. Candle Shape
    d["body_len"] = (d["close"] - d["open"]).abs() / d["open"]
    d["upper_shadow"] = (d["high"] - np.maximum(d["close"], d["open"])) / d["close"]
    d["lower_shadow"] = (np.minimum(d["close"], d["open"]) - d["low"]) / d["close"]

    d = d.dropna().reset_index(drop=True)
    
    feature_cols = [
        "rsi", "rsi_slope", "vol_z", 
        "bb_pct", "bb_width", "roc_6",
        "body_len", "upper_shadow", "lower_shadow"
    ]
    return d, feature_cols

# ------------------------------
# Strict Labeling (The Truth)
# ------------------------------
def label_pivots_strict(df: pd.DataFrame, lookback=12, lookahead=12):
    # Only label a point if it is the strict max/min in a wide window
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    labels = np.zeros(n, dtype=int) # 0=None, 1=Buy(Low), 2=Sell(High)
    
    for i in range(lookback, n - lookahead):
        # Check Low (Buy)
        if lows[i] == np.min(lows[i-lookback : i+lookahead+1]):
            labels[i] = 1
        # Check High (Sell)
        elif highs[i] == np.max(highs[i-lookback : i+lookahead+1]):
            labels[i] = 2
            
    return labels

# ------------------------------
# Main
# ------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    args = p.parse_args()

    print("\n[1/4] Data & Feature Eng")
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
    df, features = add_features_discovery(df)
    
    # Create Targets
    labels = label_pivots_strict(df, lookback=24, lookahead=24) # ~6h window
    
    # Filter dataset to only include Reversals (1,2) and some random Normals (0) for contrast
    # But for Rule Discovery, we want to know: "Given a candle, is it a Buy Reversal?"
    
    X = df[features]
    y = labels
    
    # We will train 2 separate binary classifiers to find rules for Buy and Sell separately
    # This makes the rules cleaner.
    
    out_dir = f"./all_models/models_v14/{args.symbol}"
    _safe_mkdir(os.path.join(out_dir, "plots"))
    
    rule_file_path = os.path.join(out_dir, "discovered_rules.txt")
    with open(rule_file_path, "w") as f:
        f.write(f"=== Rule Discovery Report for {args.symbol} {args.interval} ===\n")
        f.write("Method: Decision Tree (Depth=3) extraction on strict pivots (+/- 24 candles)\n\n")

    print("[2/4] Discovering BUY Rules (Pivot Lows)")
    y_buy = (y == 1).astype(int) # 1 if Buy, 0 otherwise
    
    # Use class_weight='balanced' because Buys are rare
    dt_buy = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    dt_buy.fit(X, y_buy)
    
    # Visualize Tree
    plt.figure(figsize=(20,10))
    plot_tree(dt_buy, feature_names=features, class_names=["Wait", "BUY"], filled=True, fontsize=10)
    plt.title("Decision Tree Logic for BUY Signals")
    plt.savefig(os.path.join(out_dir, "plots", "tree_buy.png"))
    plt.close()
    
    # Extract Rules
    from sklearn.tree import _tree
    def tree_to_code(tree, feature_names, class_label):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        rules = []
        def recurse(node, depth, path_str):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                recurse(tree_.children_left[node], depth + 1, path_str + [f"{name} <= {threshold:.2f}"])
                recurse(tree_.children_right[node], depth + 1, path_str + [f"{name} > {threshold:.2f}"])
            else:
                # Leaf node
                # Check if this leaf predicts the target class (1)
                # value is [[count_0, count_1]]
                counts = tree_.value[node][0]
                prob = counts[1] / (counts[0] + counts[1])
                if prob > 0.6: # High confidence leaf
                    rules.append(f"IF {' AND '.join(path_str)} THEN {class_label} (Prob: {prob:.2f})")
        
        recurse(0, 1, [])
        return rules

    buy_rules = tree_to_code(dt_buy, features, "BUY")
    with open(rule_file_path, "a") as f:
        f.write("--- Discovered BUY Rules ---\n")
        if not buy_rules: f.write("No high-confidence simple rules found (try deeper tree).\n")
        for r in buy_rules:
            f.write(r + "\n")
        f.write("\n")

    print("[3/4] Discovering SELL Rules (Pivot Highs)")
    y_sell = (y == 2).astype(int)
    dt_sell = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    dt_sell.fit(X, y_sell)
    
    plt.figure(figsize=(20,10))
    plot_tree(dt_sell, feature_names=features, class_names=["Wait", "SELL"], filled=True, fontsize=10)
    plt.title("Decision Tree Logic for SELL Signals")
    plt.savefig(os.path.join(out_dir, "plots", "tree_sell.png"))
    plt.close()
    
    sell_rules = tree_to_code(dt_sell, features, "SELL")
    with open(rule_file_path, "a") as f:
        f.write("--- Discovered SELL Rules ---\n")
        if not sell_rules: f.write("No high-confidence simple rules found.\n")
        for r in sell_rules:
            f.write(r + "\n")
        f.write("\n")
        
    print("[4/4] Feature Importance Ranking (XGBoost)")
    # Train XGBoost on Multi-class (0,1,2) to see overall importance
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    xgb.fit(X, y)
    
    plt.figure(figsize=(10, 6))
    # plot_importance(xgb, max_num_features=10) # default plotting is ugly
    # Custom plot
    importances = xgb.feature_importances_
    indices = np.argsort(importances)
    plt.title('Feature Importances (What drives Reversals?)')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", "feature_importance.png"))
    
    with open(rule_file_path, "a") as f:
        f.write("--- Top Drivers of Reversals (XGBoost) ---\n")
        for i in reversed(indices):
            f.write(f"{features[i]}: {importances[i]:.4f}\n")
            
    print(f"Done. Rules saved to {rule_file_path}")
    print(f"Check {rule_file_path} to see the 'Formula' found.")

if __name__ == "__main__":
    main()
