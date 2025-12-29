#!/usr/bin/env python3
"""colab_workflow_v26.py

V26 "The AI Trader" (LightGBM Model)

Goal: Switch from GP (Formula Finding) to ML (Pattern Recognition) to break accuracy ceiling.
Strategy:
1. Generate rich features (Lags, Rolling stats).
2. Train a LightGBM Classifier to predict 'Trend Start'.
3. Output Feature Importance to understand what matters.

Run on Colab:
!pip install lightgbm pandas numpy scikit-learn && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v26.py | python3 - \
  --symbol BTCUSDT --interval 15m
"""

import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------------
# 1. Feature Engineering (ML Style)
# ------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    # 1. Basic Indicators
    d['rsi'] = 100 - (100 / (1 + d['close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / (-d['close'].diff().clip(upper=0).ewm(alpha=1/14).mean() + 1e-12)))
    
    d['ema12'] = d['close'].ewm(span=12, adjust=False).mean()
    d['ema26'] = d['close'].ewm(span=26, adjust=False).mean()
    d['macd'] = d['ema12'] - d['ema26']
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist'] = d['macd'] - d['macd_signal']
    
    d['atr'] = pd.concat([d['high']-d['low'], (d['high']-d['close'].shift()).abs(), (d['low']-d['close'].shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    
    # 2. Rolling Stats (The "Context")
    for window in [20, 50]:
        d[f'sma_{window}'] = d['close'].rolling(window).mean()
        d[f'std_{window}'] = d['close'].rolling(window).std()
        d[f'z_score_{window}'] = (d['close'] - d[f'sma_{window}']) / (d[f'std_{window}'] + 1e-12)
    
    # 3. Lag Features (The "History")
    lags = [1, 2, 3, 5, 8]
    for col in ['close', 'volume', 'rsi', 'macd_hist', 'atr']:
        for lag in lags:
            d[f'{col}_lag_{lag}'] = d[col].shift(lag)
            if col in ['close', 'volume']:
                # For price/vol, use pct_change instead of raw value
                d[f'{col}_chg_{lag}'] = d[col].pct_change(lag)

    # 4. Price Action
    d['body_size'] = (d['close'] - d['open']) / d['open']
    d['upper_shadow'] = (d['high'] - d[['close', 'open']].max(axis=1)) / d['open']
    d['lower_shadow'] = (d[['close', 'open']].min(axis=1) - d['low']) / d['open']

    d = d.dropna().reset_index(drop=True)
    return d

# ------------------------------
# 2. Target Labeling
# ------------------------------
def label_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Trend Start: Future 20-bar return > 2.5% AND Max Drawdown < 1.0%
    future_window = 20
    min_return = 0.025
    max_drawdown = 0.01
    
    targets = np.zeros(len(df), dtype=int)
    
    for i in range(len(df) - future_window):
        entry_price = df["close"].iloc[i]
        future_prices = df["close"].iloc[i+1 : i+1+future_window]
        
        max_price = future_prices.max()
        min_price = future_prices.min()
        
        ret = (max_price - entry_price) / entry_price
        dd = (entry_price - min_price) / entry_price
        
        if ret >= min_return and dd <= max_drawdown:
            targets[i] = 1
            
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
    print("Labeling Targets...")
    df = label_targets(df)
    
    # Prepare Data for LightGBM
    # Force drop problem columns
    bad_cols = ['open_time', 'close_time', 'ignore', 'target']
    # Select only numeric types
    feature_cols = [c for c in df.columns if c not in bad_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    # Verify dtypes
    X = df[feature_cols].copy()
    # Fill remaining NaNs with 0 to be safe
    X = X.fillna(0)
    
    # Convert to float32 explicitly to catch any lingering object types
    try:
        X = X.astype(np.float32)
    except ValueError as e:
        print(f"Error converting to float32: {e}")
        # Find non-numeric columns
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                print(f"Bad column: {col} type: {X[col].dtype}")
        return

    y = df['target']
    
    print(f"Training with {X.shape[1]} features...")
    
    # Split: Train (80%), Test (20%) - Walk Forward Split (No Shuffle!)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Handle Class Imbalance (Trends are rare)
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"Positive Class Weight: {pos_weight:.2f}")
    
    print("Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        scale_pos_weight=pos_weight, # Vital for imbalanced data
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='logloss', 
              callbacks=[lgb.early_stopping(stopping_rounds=50)])
    
    print("\n--- Evaluation on TEST SET ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    
    # Filtered Performance (High Confidence only)
    threshold = 0.8
    high_conf_idx = y_prob > threshold
    if sum(high_conf_idx) > 0:
        prec = precision_score(y_test[high_conf_idx], (y_prob[high_conf_idx] > 0.5).astype(int))
        print(f"\n[High Confidence Filter > {threshold}]")
        print(f"Trades: {sum(high_conf_idx)}")
        print(f"Precision: {prec:.4f}")
    
    # Feature Importance
    print("\n--- Top 10 Important Features ---")
    imp = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    print(imp.sort_values('importance', ascending=False).head(10))
    
    # Save model
    out_dir = f"./all_models/models_v26/{args.symbol}"
    _safe_mkdir(out_dir)
    model.booster_.save_model(os.path.join(out_dir, "lgbm_model.txt"))
    print(f"\nModel saved to {out_dir}")

if __name__ == "__main__":
    main()
