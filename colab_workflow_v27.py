#!/usr/bin/env python3
"""colab_workflow_v27.py

V27 "Trend School" (Relaxed Targets)

Goal: Fix V26's "Class Collapse" by relaxing the target definition.
Strategy:
1. Target: Future 12-bar (3h) return > 1.0% (Easier to learn).
2. Hyperparameters: Lower learning rate, smaller max_depth to prevent overfitting to noise.
3. Class Weight: Balanced.

Run on Colab:
!pip install lightgbm pandas numpy scikit-learn && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v27.py | python3 - \
  --symbol BTCUSDT --interval 15m
"""

import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------------
# 1. Feature Engineering (Standard ML)
# ------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    # Basic Indicators
    d['rsi'] = 100 - (100 / (1 + d['close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / (-d['close'].diff().clip(upper=0).ewm(alpha=1/14).mean() + 1e-12)))
    
    d['ema12'] = d['close'].ewm(span=12, adjust=False).mean()
    d['ema26'] = d['close'].ewm(span=26, adjust=False).mean()
    d['macd'] = d['ema12'] - d['ema26']
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist'] = d['macd'] - d['macd_signal']
    
    d['atr'] = pd.concat([d['high']-d['low'], (d['high']-d['close'].shift()).abs(), (d['low']-d['close'].shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    
    # Rolling Stats
    for window in [12, 24, 96]: # 3h, 6h, 24h
        d[f'sma_{window}'] = d['close'].rolling(window).mean()
        d[f'std_{window}'] = d['close'].rolling(window).std()
        d[f'z_score_{window}'] = (d['close'] - d[f'sma_{window}']) / (d[f'std_{window}'] + 1e-12)
        d[f'roc_{window}'] = d['close'].pct_change(window)
    
    # Lag Features
    lags = [1, 3, 5]
    for col in ['rsi', 'macd_hist', 'volume']:
        for lag in lags:
            d[f'{col}_lag_{lag}'] = d[col].shift(lag)

    d = d.dropna().reset_index(drop=True)
    return d

# ------------------------------
# 2. Target Labeling (RELAXED)
# ------------------------------
def label_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Trend Start V27: 
    # Just Price rises > 1.0% in next 12 bars (3 hours)
    # No drawdown constraint for now (let it learn volatility later)
    
    future_window = 12
    min_return = 0.010 # 1.0%
    
    targets = np.zeros(len(df), dtype=int)
    
    # Vectorized calculation for speed
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
    
    print("Generating Features...")
    df = add_features(df)
    print("Labeling Targets (Relaxed)...")
    df = label_targets(df)
    
    # Prepare Data
    bad_cols = ['open_time', 'close_time', 'ignore', 'target']
    feature_cols = [c for c in df.columns if c not in bad_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols].copy().fillna(0).astype(np.float32)
    y = df['target']
    
    print(f"Training with {X.shape[1]} features...")
    print(f"Positive Samples: {sum(y)} / {len(y)} ({sum(y)/len(y):.2%})")
    
    # Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print("Training LightGBM (Relaxed)...")
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01, # Slower learning
        max_depth=4,        # Shallower trees to prevent overfitting
        num_leaves=15,
        class_weight='balanced', # Crucial
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', 
              callbacks=[lgb.early_stopping(stopping_rounds=100)])
    
    print("\n--- Evaluation on TEST SET ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # High Confidence Filter
    thresholds = [0.6, 0.7, 0.8]
    for thresh in thresholds:
        mask = y_prob > thresh
        if sum(mask) > 0:
            p = precision_score(y_test[mask], (y_prob[mask] > 0.5).astype(int))
            print(f"[Conf > {thresh:.1f}] Trades: {sum(mask)}, Precision: {p:.4f}")
        else:
            print(f"[Conf > {thresh:.1f}] No trades.")

    # Feature Importance
    print("\n--- Top 10 Important Features ---")
    imp = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    print(imp.sort_values('importance', ascending=False).head(10))
    
    # Save model
    out_dir = f"./all_models/models_v27/{args.symbol}"
    _safe_mkdir(out_dir)
    model.booster_.save_model(os.path.join(out_dir, "lgbm_model.txt"))

if __name__ == "__main__":
    main()
