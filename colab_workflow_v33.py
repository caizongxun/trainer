#!/usr/bin/env python3
"""colab_workflow_v33.py

V33 "The Optimizer" (Automated Hyperparameter Tuning with Optuna)

Goal: Replace manual guessing with Bayesian Optimization to find the perfect LightGBM hyperparameters.
Strategy:
1. Use Optuna to define a search space (learning_rate, num_leaves, regularization, etc.).
2. Run 50 trials to maximize ROC AUC on a validation set.
3. Train the final model with the best parameters found.

Run on Colab:
!pip install optuna lightgbm pandas numpy scikit-learn && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v33.py | python3 - \
  --symbol BTCUSDT --interval 15m
"""

import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import classification_report, precision_score, roc_auc_score

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------------
# 1. Feature Engineering (The Solid V32 Set)
# ------------------------------
def get_indicators(df):
    d = df.copy()
    d['vol_squeeze'] = d['close'].rolling(20).std() / (d['close'].rolling(96).std() + 1e-9)
    d['ema_50'] = d['close'].ewm(span=50, adjust=False).mean()
    d['trend_strength'] = (d['ema_50'] - d['ema_50'].shift(5)) / d['ema_50'].shift(5)
    bb_mean = d['close'].rolling(20).mean()
    bb_std = d['close'].rolling(20).std()
    d['pct_b'] = (d['close'] - (bb_mean - 2*bb_std)) / (4*bb_std + 1e-9)
    d['rsi'] = 100 - (100 / (1 + d['close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / (-d['close'].diff().clip(upper=0).ewm(alpha=1/14).mean() + 1e-12)))
    return d

def add_features_v33(df_15m: pd.DataFrame) -> pd.DataFrame:
    df_15m = get_indicators(df_15m)
    df_4h = df_15m.set_index('open_time').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_4h = get_indicators(df_4h)
    df_4h = df_4h.add_suffix('_4h')
    
    df_15m = df_15m.set_index('open_time')
    df_15m = pd.merge_asof(df_15m.sort_index(), df_4h.sort_index(), left_index=True, right_index=True, direction='backward')
    df_15m = df_15m.reset_index().dropna()
    
    d = df_15m
    d['dip_quality'] = d['trend_strength_4h'] * (1 - d['pct_b']) 
    return d

# ------------------------------
# 2. Target Labeling
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
    
    print("Feature Engineering (Optuna Ready)...")
    df = add_features_v33(df)
    print("Labeling Targets...")
    df = label_targets(df)
    
    bad_cols = ['open_time', 'close_time', 'ignore', 'target']
    feature_cols = [c for c in df.columns if c not in bad_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols].copy().fillna(0).astype(np.float32)
    y = df['target']
    
    # Split: Train/Val/Test
    # Train (60%), Val (20% for Optuna), Test (20% for Final Eval)
    n = len(df)
    train_idx = int(n * 0.6)
    val_idx = int(n * 0.8)
    
    X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
    X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
    X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # ------------------------------
    # 3. Optuna Optimization
    # ------------------------------
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'n_jobs': -1
        }
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)
        
        gbm = lgb.train(param, dtrain, valid_sets=[dval], callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)])
        preds = gbm.predict(X_val)
        auc = roc_auc_score(y_val, preds)
        return auc

    print("Starting Optuna Study (50 Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # ------------------------------
    # 4. Final Training with Best Params
    # ------------------------------
    print("\nTraining Final Model with Best Params...")
    best_params = trial.params
    best_params['n_estimators'] = 5000 # Give it plenty of room
    best_params['random_state'] = 42
    
    # Merge Train + Val for final training
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train_full, y_train_full, eval_set=[(X_test, y_test)], eval_metric='auc',
              callbacks=[lgb.early_stopping(stopping_rounds=100)])
    
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
    
    out_dir = f"./all_models/models_v33/{args.symbol}"
    _safe_mkdir(out_dir)
    model.booster_.save_model(os.path.join(out_dir, "lgbm_model.txt"))

if __name__ == "__main__":
    main()
