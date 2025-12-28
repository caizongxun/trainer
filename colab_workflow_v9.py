#!/usr/bin/env python3
"""colab_workflow_v9.py

V9.5 "Zero-Lag" Crypto Forecast Trainer

Key goals:
- Reduce lag by punishing directional errors (wrong sign) more heavily.
- Add "velocity" and "acceleration" features (ROC, MACD Histogram) to help pre-empt turns.
- Revert to Huber loss for sharper reaction than Log-Cosh, but keep Tanh clip for safety.
- Explicitly penalize lag: Loss = Huber + 2.0 * Direction_Error.

Workflow:
  1) env setup
  2) fetch data
  3) train
  4) save

Colab (GPU) one-liner:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v9.py | python3 -

Run with:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v9.py | python3 - \
  --symbol BTCUSDT --interval 15m --epochs 100 --time_budget_min 120 --upload 0
"""

import os
import re
import gc
import json
import time
import math
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def _print_step(step: str, msg: str) -> None:
    print(f"\n[{step}] {msg}")


def _print_kv(k: str, v) -> None:
    print(f"  - {k}: {v}")


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _configure_tf() -> dict:
    info = {
        "tf_version": tf.__version__,
        "num_gpus": len(tf.config.list_physical_devices("GPU")),
        "mixed_precision": False,
        "xla": False,
    }
    
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        info["mixed_precision"] = True
    except:
        pass

    try:
        tf.config.optimizer.set_jit(True)
        info["xla"] = True
    except:
        pass

    return info


# ------------------------------
# Feature engineering (V9.5 Enhanced)
# ------------------------------

def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def add_price_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]

    time_col = next((c for c in ["open_time", "opentime", "timestamp", "time", "date"] if c in d.columns), None)
    if not time_col:
        raise ValueError("Missing time column")

    if pd.api.types.is_numeric_dtype(d[time_col]):
        ts = d[time_col].astype("int64")
        unit = "ms" if ts.median() > 10**12 else "s"
        d["open_time"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        d["open_time"] = pd.to_datetime(d[time_col], utc=True, errors="coerce")

    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna().sort_values("open_time").reset_index(drop=True)

    # 1. Base Log returns
    d["log_ret"] = np.log(d["close"] / d["close"].shift(1))
    
    # 2. Shadows and ranges
    d["upper_shadow"] = (d["high"] - np.maximum(d["close"], d["open"])) / d["close"]
    d["lower_shadow"] = (np.minimum(d["close"], d["open"]) - d["low"]) / d["close"]
    d["body"] = (d["close"] - d["open"]) / d["open"]
    
    # 3. RSI
    d["rsi"] = _wilder_rsi(d["close"]) / 100.0
    
    # 4. Volume
    d["log_vol"] = np.log1p(d["volume"])
    d["log_vol_diff"] = d["log_vol"].diff()

    # 5. Time
    d["hour_sin"] = np.sin(2 * np.pi * d["open_time"].dt.hour / 24.0)
    d["dow_sin"] = np.sin(2 * np.pi * d["open_time"].dt.dayofweek / 7.0)

    # 6. Moving Averages distance (Trend)
    for span in [7, 25, 99]:
        ema = d["close"].ewm(span=span, adjust=False).mean()
        d[f"dist_ema{span}"] = np.log(d["close"] / ema)

    # 7. V9.5 New: Momentum & Acceleration Features
    # ROC (Rate of Change) - Velocity
    d["roc_6"] = d["close"].pct_change(6)
    d["roc_12"] = d["close"].pct_change(12)
    
    # MACD Histogram - Acceleration (detects turns before they happen)
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = macd - signal  # This is a leading indicator for momentum shifts

    d = d.dropna().reset_index(drop=True)

    feature_cols = [
        "log_ret", "upper_shadow", "lower_shadow", "body", 
        "rsi", "log_vol", "log_vol_diff", "hour_sin", "dow_sin",
        "dist_ema7", "dist_ema25", "dist_ema99",
        "roc_6", "roc_12", "macd_hist"
    ]
    return d, feature_cols


# ------------------------------
# Data pipeline
# ------------------------------

def create_sequences(df: pd.DataFrame, cols: list[str], seq_len: int, pred_len: int):
    feat = df[cols].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)
    
    n = len(df)
    max_i = n - seq_len - pred_len
    
    X = np.zeros((max_i, seq_len, len(cols)), dtype=np.float32)
    y = np.zeros((max_i, pred_len), dtype=np.float32)
    base_close = np.zeros((max_i,), dtype=np.float32)
    
    for i in range(max_i):
        X[i] = feat[i : i + seq_len]
        base_idx = i + seq_len - 1
        bc = close[base_idx]
        base_close[i] = bc
        
        # Target: Log return of future close vs base close
        future_close = close[i + seq_len : i + seq_len + pred_len]
        y[i] = np.log(future_close / bc)

    return X, y, base_close, np.arange(seq_len + pred_len - 1, n - 1)[:max_i]


# ------------------------------
# Model V9.5 Zero-Lag
# ------------------------------

def build_model_v9_5(seq_len: int, n_feat: int, pred_len: int, lr: float):
    from tensorflow.keras import layers, Model
    
    inp = layers.Input(shape=(seq_len, n_feat))
    
    # 1. Conv1D for local pattern extraction
    x = layers.Conv1D(64, 3, activation="swish", padding="same")(inp)
    x = layers.MaxPooling1D(2)(x)
    
    # 2. Bidirectional LSTM for sequence context
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.2))(x)
    
    # 3. Dense Head
    x = layers.Dense(128, activation="swish")(x)
    x = layers.Dropout(0.2)(x)
    
    # 4. Tanh Output Limiter (Safety first)
    raw_out = layers.Dense(pred_len)(x)
    out = layers.Activation("tanh")(raw_out) 
    out = layers.Rescaling(scale=0.1)(out) 
    
    model = Model(inp, out, name="v9.5_zero_lag")

    # V9.5 Custom Loss: Directional Penalty
    def zero_lag_loss(y_true, y_pred):
        # 1. Standard Huber loss (robust regression)
        huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
        
        # 2. Directional Penalty
        # If signs match (product > 0), penalty is 0.
        # If signs differ (product < 0), penalty is |y_pred|.
        # We punish the model for predicting a move in the WRONG direction.
        sign_mismatch = tf.where(tf.math.multiply(y_true, y_pred) < 0, tf.abs(y_pred - y_true), 0.0)
        dir_penalty = tf.reduce_mean(sign_mismatch)
        
        return huber + 2.0 * dir_penalty

    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-3)
    model.compile(optimizer=opt, loss=zero_lag_loss, metrics=["mae"])
    
    return model


# ------------------------------
# Workflow
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="15m")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--time_budget_min", type=int, default=120)
    parser.add_argument("--upload", type=int, default=0)
    parser.add_argument("--hf_token", type=str, default="")
    args = parser.parse_args()

    # Fixed hyperparams
    SEQ_LEN = 96
    PRED_LEN = 10
    
    _print_step("1/5", "Setup")
    _configure_tf()
    _set_seed(42)
    
    _print_step("2/5", "Data")
    from huggingface_hub import snapshot_download
    
    # Robust download logic
    path = snapshot_download(
        repo_id="zongowo111/cpb-models", 
        repo_type="dataset",
        allow_patterns=None,
        ignore_patterns=None
    )
    
    csv_file = None
    all_files_found = []
    for root, _, files in os.walk(path):
        for f in files:
            all_files_found.append(f)
            if f.endswith(".csv") and args.symbol in f and args.interval in f:
                csv_file = os.path.join(root, f)
                break
        if csv_file:
            break
            
    if not csv_file:
        print(f"DEBUG: Found {len(all_files_found)} files. First 20: {all_files_found[:20]}")
        raise ValueError(f"No CSV found for {args.symbol} {args.interval}")
        
    _print_kv("csv", csv_file)
    
    df = pd.read_csv(csv_file)
    df, features = add_price_features(df)
    
    train_size = int(len(df) * 0.9)
    scaler = StandardScaler()
    df.loc[:train_size-1, features] = scaler.fit_transform(df.loc[:train_size-1, features])
    df.loc[train_size:, features] = scaler.transform(df.loc[train_size:, features])
    
    X, y, base_close, idxs = create_sequences(df, features, SEQ_LEN, PRED_LEN)
    
    train_mask = idxs < train_size
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val, b_val = X[~train_mask], y[~train_mask], base_close[~train_mask]
    
    _print_step("3/5", "Train V9.5")
    model = build_model_v9_5(SEQ_LEN, len(features), PRED_LEN, 1e-3)
    
    class ValPriceMAE(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 2 == 0:
                pred_log = self.model.predict(X_val[:1000], verbose=0)
                pred_p = b_val[:1000, None] * np.exp(pred_log)
                true_p = b_val[:1000, None] * np.exp(y_val[:1000])
                mape = np.mean(np.abs((true_p - pred_p) / true_p))
                logs["val_mape"] = mape
                print(f" - val_mape: {mape:.4f}")

    ckpt_path = f"best_model_{args.symbol}.keras"
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ValPriceMAE()
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=64,
        callbacks=cb,
        verbose=1
    )
    
    out_dir = f"./all_models/models_v9/{args.symbol}"
    _safe_mkdir(os.path.join(out_dir, "plots"))
    model.save(os.path.join(out_dir, f"{args.symbol}_{args.interval}_v9.keras"))
    
    import matplotlib.pyplot as plt
    pred_log = model.predict(X_val, verbose=0)
    pred_p = b_val[:, None] * np.exp(pred_log)
    true_p = b_val[:, None] * np.exp(y_val)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(true_p[:, 0], label="True")
    plt.plot(pred_p[:, 0], label="Pred", alpha=0.8)
    plt.title(f"{args.symbol} {args.interval} Horizon=1 (Zero-Lag)")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(true_p[:, -1], label="True")
    plt.plot(pred_p[:, -1], label="Pred", alpha=0.8)
    plt.title(f"{args.symbol} {args.interval} Horizon={PRED_LEN} (Zero-Lag)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", "forecast.png"))
    print(f"Done. Saved to {out_dir}")

if __name__ == "__main__":
    main()
