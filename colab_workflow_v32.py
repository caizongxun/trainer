#!/usr/bin/env python3
"""colab_workflow_v32.py

V32 "Unchained Sniper"

Goal: Recover from V31's "Constraint Paralysis".
Analysis: The strict monotone constraints in V30/V31 forced the model into a corner where no data point satisfied all conditions perfectly, leading to 0 trades.
Strategy:
1. Remove Monotone Constraints: Let the model learn non-linear relationships freely (e.g., maybe high volatility IS good sometimes).
2. Keep the Feature Set: V31's features (MTF + Location) are solid.
3. Optimize Hyperparameters: Increase depth slightly to capture complex interactions without constraints.

Run on Colab:
!pip install lightgbm pandas numpy scikit-learn && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v32.py | python3 - \
  --symbol BTCUSDT --interval 15m
"""

import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, precision_score, roc_auc_score

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------------
# 1. Feature Engineering (Same as V31 - Solid Logic)
# ------------------------------
def get_indicators(df):
    d = df.copy()
    
    # 1. Volatility Squeeze
    d['vol_squeeze'] = d['close'].rolling(20).std() / (d['close'].rolling(96).std() + 1e-9)
    
    # 2. Trend (Long term)
    d['ema_50'] = d['close'].ewm(span=50, adjust=False).mean()
    d['trend_strength'] = (d['ema_50'] - d['ema_50'].shift(5)) / d['ema_50'].shift(5)
    
    # 3. Location (Cheap or Expensive?)
    bb_mean = d['close'].rolling(20).mean()
    bb_std = d['close'].rolling(20).std()
    d['pct_b'] = (d['close'] - (bb_mean - 2*bb_std)) / (4*bb_std + 1e-9)
    
    d['rsi'] = 100 - (100 / (1 + d['close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / (-d['close'].diff().clip(upper=0).ewm(alpha=1/14).mean() + 1e-12)))
    
    return d

def add_features_v32(df_15m: pd.DataFrame) -> pd.DataFrame:
    df_15m = get_indicators(df_15m)
    
    df_4h = df_15m.set_index('open_time').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_4h = get_indicators(df_4h)
    df_4h = df_4h.add_suffix('_4h')
    
    df_15m = df_15m.set_index('open_time')
    df_15m = pd.merge_asof(df_15m.sort_index(), df_4h.sort_index(), left_index=True, right_index=True, direction='backward')
    df_15m = df_15m.reset_index().dropna()
    
    # Explicit Interaction
    d = df_15m
    d['dip_quality'] = d['trend_strength_4h'] * (1 - d['pct_b']) 
    
    return d

# ------------------------------
# 2. Target Labeling (Same)
# ------------------------------
def label_targets(df: pd.DataFrame) -> pd.DataFrame:
    future_window = 12
    min_return = 0.012 
    
    targets = np.zeros(len(df), dtype=int)
    future_max = df['close'].rolling(future_window).max().shift(-future_window)
    ret = (future_max - df['close']) / df['close']
    
    targets = (ret >= min_return).astype(int)
    df["target"] = targets
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    args = p.parse_args()
    
    print(f"Loading data for {args.symbol} {args.interval}...")
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="zongowo111/cpb-models", repo_type="dataset")
    csv_file = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv") and args.symbol in f and args.interval in f:
                csv_file = os.path.join(root, f)
                break
    
    df = pd.read_csv(csv_file)
    time_col = next(c for c in df.columns if "time" in c)
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df["open_time"] = pd.to_datetime(df[time_col], unit="ms", utc=True)
    else:
        df["open_time"] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    
    print("Feature Engineering (Unchained)...")
    df = add_features_v32(df)
    print("Labeling Targets...")
    df = label_targets(df)
    
    bad_cols = ['open_time', 'close_time', 'ignore', 'target']
    feature_cols = [c for c in df.columns if c not in bad_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols].copy().fillna(0).astype(np.float32)
    y = df['target']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Positive Samples: {sum(y)} / {len(y)} ({sum(y)/len(y):.2%})")
    
    # Weight: Back to 'balanced' to encourage finding ANY signal
    # We rely on post-prediction thresholding for precision
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print("Training LightGBM (Unchained)...")
    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=7,  # Deeper trees
        num_leaves=63, # More leaves
        class_weight='balanced', # Force it to learn the minority class
        subsample=0.8,
        colsample_bytree=0.8,
        # NO MONOTONE CONSTRAINTS
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', 
              callbacks=[lgb.early_stopping(stopping_rounds=150)])
    
    print("\n--- Evaluation on TEST SET ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Confidence Buckets
    print("\n--- Confidence Analysis ---")
    for thresh in [0.6, 0.7, 0.8, 0.9]:
        mask = y_prob > thresh
        count = sum(mask)
        if count > 0:
            p = precision_score(y_test[mask], (y_prob[mask] > 0.5).astype(int))
            print(f"[Conf > {thresh:.1f}] Trades: {count}, Precision: {p:.4f}")
        else:
            print(f"[Conf > {thresh:.1f}] No trades.")

    # Feature Importance
    print("\n--- Top 10 Features ---")
    imp = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    print(imp.sort_values('importance', ascending=False).head(10))
    
    # Save model
    out_dir = f"./all_models/models_v32/{args.symbol}"
    _safe_mkdir(out_dir)
    model.booster_.save_model(os.path.join(out_dir, "lgbm_model.txt"))

if __name__ == "__main__":
    main()
