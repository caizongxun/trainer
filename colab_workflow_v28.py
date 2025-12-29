#!/usr/bin/env python3
"""colab_workflow_v28.py

V28 "Volatility Sniper"

Goal: Capitalize on V27's insight that 'std_96' (Volatility) is the #1 predictor.
Strategy:
1. Feature Engineering: Focus on Volatility Squeeze & Expansion (BB Width, Keltner Channels).
2. Model: LightGBM with Monotone Constraints to enforce logic (e.g., Higher breakout momentum = Higher score).
3. Target: Maintain V27's relaxed target but with a slight twist for quality.

Run on Colab:
!pip install lightgbm pandas numpy scikit-learn && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v28.py | python3 - \
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
# 1. Feature Engineering (Volatility Focus)
# ------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    # 1. Bollinger Bands (The Volatility King)
    for window in [20, 96]: # Short term & Daily
        sma = d['close'].rolling(window).mean()
        std = d['close'].rolling(window).std()
        d[f'bb_upper_{window}'] = sma + (2 * std)
        d[f'bb_lower_{window}'] = sma - (2 * std)
        # BB Width: Narrow = Squeeze (Potential breakout), Wide = Volatile
        d[f'bb_width_{window}'] = (d[f'bb_upper_{window}'] - d[f'bb_lower_{window}']) / sma
        # %B: Where is price relative to bands?
        d[f'bb_pct_b_{window}'] = (d['close'] - d[f'bb_lower_{window}']) / (d[f'bb_upper_{window}'] - d[f'bb_lower_{window}'] + 1e-9)

    # 2. ATR & Normalized ATR (NATR)
    tr = pd.concat([d['high']-d['low'], (d['high']-d['close'].shift()).abs(), (d['low']-d['close'].shift()).abs()], axis=1).max(axis=1)
    d['atr14'] = tr.rolling(14).mean()
    d['natr14'] = (d['atr14'] / d['close']) * 100 # Normalized volatility
    
    # 3. Volatility Ratios (Compression Detection)
    # Is current volatility lower than long-term volatility? (Squeeze)
    d['vol_squeeze'] = d['close'].rolling(20).std() / (d['close'].rolling(96).std() + 1e-9)
    
    # 4. Momentum (Direction)
    d['rsi'] = 100 - (100 / (1 + d['close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / (-d['close'].diff().clip(upper=0).ewm(alpha=1/14).mean() + 1e-12)))
    d['macd'] = d['close'].ewm(span=12).mean() - d['close'].ewm(span=26).mean()
    
    # 5. Volume confirming Volatility
    d['vol_norm'] = d['volume'] / (d['volume'].rolling(50).mean() + 1e-9)

    d = d.dropna().reset_index(drop=True)
    return d

# ------------------------------
# 2. Target Labeling (Trend Start)
# ------------------------------
def label_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Target: 3-hour return > 1.2% (Slightly harder than V27 to filter noise)
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
    
    print("Feature Engineering (Volatility Focused)...")
    df = add_features(df)
    print("Labeling Targets...")
    df = label_targets(df)
    
    # Prepare Data
    bad_cols = ['open_time', 'close_time', 'ignore', 'target']
    feature_cols = [c for c in df.columns if c not in bad_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols].copy().fillna(0).astype(np.float32)
    y = df['target']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Positive Samples: {sum(y)} / {len(y)} ({sum(y)/len(y):.2%})")
    
    # Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print("Training LightGBM (Volatility Sniper)...")
    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.005, # Even slower for precision
        max_depth=4,
        num_leaves=15,
        class_weight='balanced',
        subsample=0.8,       # Bagging to reduce variance
        colsample_bytree=0.8, # Feature subsampling
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
    print("\n--- Top 10 Features (Volatility Check) ---")
    imp = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    print(imp.sort_values('importance', ascending=False).head(10))
    
    # Save model
    out_dir = f"./all_models/models_v28/{args.symbol}"
    _safe_mkdir(out_dir)
    model.booster_.save_model(os.path.join(out_dir, "lgbm_model.txt"))

if __name__ == "__main__":
    main()
