#!/usr/bin/env python3
"""colab_workflow_v30.py

V30 "The Disciplined Sniper"

Goal: Combine V29's High AUC (Context) with V28's High Precision (Squeeze).
Strategy:
1. Feature Selection: Remove 'dist_ema50' to prevent trend-chasing.
2. Interaction Features: Explicitly create 'Trend * Squeeze' features.
3. Monotone Constraints: FORCE the model to prefer lower volatility (Squeeze) for higher scores.
   - Constraint: vol_squeeze must have negative correlation with target probability.

Run on Colab:
!pip install lightgbm pandas numpy scikit-learn && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v30.py | python3 - \
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
# 1. Feature Engineering (MTF + Interactions)
# ------------------------------
def get_indicators(df):
    d = df.copy()
    # Volatility Squeeze (The Key)
    # Ratio < 1.0 means current volatility is lower than long-term (Squeezed)
    d['vol_squeeze'] = d['close'].rolling(20).std() / (d['close'].rolling(96).std() + 1e-9)
    
    # Trend Strength (ADX-like proxy using EMAs)
    d['ema_short'] = d['close'].ewm(span=12).mean()
    d['ema_long'] = d['close'].ewm(span=50).mean()
    d['trend_strength'] = (d['ema_short'] - d['ema_long']) / d['ema_long'] # Normalized MACD
    
    # RSI for momentum
    d['rsi'] = 100 - (100 / (1 + d['close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / (-d['close'].diff().clip(upper=0).ewm(alpha=1/14).mean() + 1e-12)))
    
    return d

def add_features_v30(df_15m: pd.DataFrame) -> pd.DataFrame:
    # 1. Base 15m Features
    df_15m = get_indicators(df_15m)
    
    # 2. Resample to 4H only (Simplifying context to just the major trend)
    df_4h = df_15m.set_index('open_time').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_4h = get_indicators(df_4h)
    df_4h = df_4h.add_suffix('_4h')
    
    # 3. Merge
    df_15m = df_15m.set_index('open_time')
    df_15m = pd.merge_asof(df_15m.sort_index(), df_4h.sort_index(), left_index=True, right_index=True, direction='backward')
    df_15m = df_15m.reset_index().dropna()
    
    # 4. Explicit Interaction (The "Sniper Setup")
    # Setup = 4H Trend is UP AND 15m Volatility is LOW
    # We want to feed raw components, but these ratios help tree splits
    d = df_15m
    d['setup_quality'] = d['trend_strength_4h'] / (d['vol_squeeze'] + 0.1) 
    
    return d

# ------------------------------
# 2. Target Labeling (Same V28/V29)
# ------------------------------
def label_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Target: 3-hour return > 1.2%
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
    
    print("Feature Engineering (Disciplined Sniper)...")
    df = add_features_v30(df)
    print("Labeling Targets...")
    df = label_targets(df)
    
    # Prepare Data
    bad_cols = ['open_time', 'close_time', 'ignore', 'target']
    feature_cols = [c for c in df.columns if c not in bad_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols].copy().fillna(0).astype(np.float32)
    y = df['target']
    
    # MONOTONE CONSTRAINTS
    # We want to force the model: Lower 'vol_squeeze' -> Higher Score
    # 0 = no constraint, -1 = decreasing, 1 = increasing
    monotone_constraints = []
    for col in feature_cols:
        if col == 'vol_squeeze':
            monotone_constraints.append(-1) # Force negative correlation
            print(f"Constraint: {col} must be decreasing (Lower is better)")
        elif col == 'trend_strength_4h':
            monotone_constraints.append(1)  # Force positive correlation (Trend following)
            print(f"Constraint: {col} must be increasing (Higher is better)")
        else:
            monotone_constraints.append(0)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Positive Samples: {sum(y)} / {len(y)} ({sum(y)/len(y):.2%})")
    
    # Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print("Training LightGBM (Constrained)...")
    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.005,
        max_depth=5,
        num_leaves=20,
        class_weight='balanced',
        subsample=0.8,
        colsample_bytree=0.8,
        monotone_constraints=monotone_constraints, # THE KEY
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
    print("\n--- Top 10 Features (Disciplined) ---")
    imp = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    print(imp.sort_values('importance', ascending=False).head(10))
    
    # Save model
    out_dir = f"./all_models/models_v30/{args.symbol}"
    _safe_mkdir(out_dir)
    model.booster_.save_model(os.path.join(out_dir, "lgbm_model.txt"))

if __name__ == "__main__":
    main()
