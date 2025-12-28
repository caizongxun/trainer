#!/usr/bin/env python3
"""colab_workflow_v9.py

V9.4 "Smooth Operator" Crypto Forecast Trainer

Key goals:
- Eliminate "spiky" predictions by removing Vol Head and using Log-Cosh loss.
- Increase seq_len to 96 (approx 24h for 15m) to capture robust trends.
- Use Tanh + scaling in output layer to physically limit single-step drift.
- Switch to StandardScaler for better handling of normal distributions.
- Simplify architecture to pure LSTM + Attention without auxiliary tasks.

Workflow:
  1) env setup
  2) fetch data
  3) train (epochs up to 100)
  4) save
  5) upload (optional)

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
# Feature engineering
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

    # Log returns
    d["log_ret"] = np.log(d["close"] / d["close"].shift(1))
    
    # Shadows and ranges
    d["upper_shadow"] = (d["high"] - np.maximum(d["close"], d["open"])) / d["close"]
    d["lower_shadow"] = (np.minimum(d["close"], d["open"]) - d["low"]) / d["close"]
    d["body"] = (d["close"] - d["open"]) / d["open"]
    
    # RSI
    d["rsi"] = _wilder_rsi(d["close"]) / 100.0
    
    # Volume
    d["log_vol"] = np.log1p(d["volume"])
    d["log_vol_diff"] = d["log_vol"].diff()

    # Time
    d["hour_sin"] = np.sin(2 * np.pi * d["open_time"].dt.hour / 24.0)
    d["dow_sin"] = np.sin(2 * np.pi * d["open_time"].dt.dayofweek / 7.0)

    # Moving Averages distance
    for span in [7, 25, 99]:
        ema = d["close"].ewm(span=span, adjust=False).mean()
        d[f"dist_ema{span}"] = np.log(d["close"] / ema)

    d = d.dropna().reset_index(drop=True)

    feature_cols = [
        "log_ret", "upper_shadow", "lower_shadow", "body", 
        "rsi", "log_vol", "log_vol_diff", "hour_sin", "dow_sin",
        "dist_ema7", "dist_ema25", "dist_ema99"
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
        # shape (pred_len,)
        future_close = close[i + seq_len : i + seq_len + pred_len]
        y[i] = np.log(future_close / bc)

    return X, y, base_close, np.arange(seq_len + pred_len - 1, n - 1)[:max_i]


# ------------------------------
# Model V9.4
# ------------------------------

def build_model_v9_4(seq_len: int, n_feat: int, pred_len: int, lr: float):
    from tensorflow.keras import layers, Model
    
    inp = layers.Input(shape=(seq_len, n_feat))
    
    # 1. Convolutional feature extraction
    x = layers.Conv1D(64, 3, activation="swish", padding="same")(inp)
    x = layers.Conv1D(64, 3, activation="swish", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)  # Downsample to see wider context
    
    # 2. LSTM context
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.25))(x)
    
    # 3. Dense Head
    x = layers.Dense(128, activation="swish")(x)
    x = layers.Dropout(0.2)(x)
    
    # 4. Tanh Output Limiter
    # Output is log-return. We constrain it to +/- 0.10 per step range to prevent explosions
    raw_out = layers.Dense(pred_len)(x)
    # Scale Tanh: max possible movement is +/- 0.10 (approx 10%) relative to base
    # This prevents the model from ever predicting a 50% crash in 10 steps
    out = layers.Activation("tanh")(raw_out) 
    out = layers.Rescaling(scale=0.1)(out) 
    
    model = Model(inp, out, name="v9.4_smooth_operator")

    # Log-Cosh Loss: Smooth L1-like loss that suppresses outliers
    def log_cosh_loss(y_true, y_pred):
        return tf.reduce_mean(tf.math.log(tf.cosh(y_true - y_pred)))
    
    # Shape loss to force smoothness across horizon
    def shape_loss(y_true, y_pred):
        dy_true = y_true[:, 1:] - y_true[:, :-1]
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        return tf.reduce_mean(tf.square(dy_true - dy_pred))

    def total_loss(y_true, y_pred):
        return log_cosh_loss(y_true, y_pred) + 5.0 * shape_loss(y_true, y_pred)

    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-3)
    model.compile(optimizer=opt, loss=total_loss, metrics=["mae"])
    
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

    # Fixed hyperparams for V9.4
    SEQ_LEN = 96  # ~24 hours for 15m
    PRED_LEN = 10
    
    _print_step("1/5", "Setup")
    _configure_tf()
    _set_seed(42)
    
    _print_step("2/5", "Data")
    from huggingface_hub import snapshot_download
    
    # Fix: snapshot_download allow_patterns is tricky.
    # We download all json/csv files to be safe, then filter locally.
    path = snapshot_download(
        repo_id="zongowo111/cpb-models", 
        repo_type="dataset",
        allow_patterns=["**/*.csv", "**/*.json"]
    )
    
    # Recursive search for the CSV file
    csv_file = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv") and args.symbol in f and args.interval in f:
                csv_file = os.path.join(root, f)
                break
        if csv_file:
            break
            
    if not csv_file:
        # Fallback: list all CSVs found to help debug
        all_csvs = [f for r,_,fs in os.walk(path) for f in fs if f.endswith(".csv")]
        print(f"DEBUG: Found CSVs: {all_csvs}")
        raise ValueError(f"No CSV found for {args.symbol} {args.interval} in {path}")
        
    _print_kv("csv", csv_file)
    
    df = pd.read_csv(csv_file)
    df, features = add_price_features(df)
    
    # Split
    train_size = int(len(df) * 0.9)
    
    # Scaler: StandardScaler (Z-score)
    scaler = StandardScaler()
    df.loc[:train_size-1, features] = scaler.fit_transform(df.loc[:train_size-1, features])
    df.loc[train_size:, features] = scaler.transform(df.loc[train_size:, features])
    
    X, y, base_close, idxs = create_sequences(df, features, SEQ_LEN, PRED_LEN)
    
    train_mask = idxs < train_size
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val, b_val = X[~train_mask], y[~train_mask], base_close[~train_mask]
    
    _print_step("3/5", "Train V9.4")
    model = build_model_v9_4(SEQ_LEN, len(features), PRED_LEN, 1e-3)
    
    # Callback to track real price error
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
    
    # Save artifacts
    out_dir = f"./all_models/models_v9/{args.symbol}"
    _safe_mkdir(os.path.join(out_dir, "plots"))
    model.save(os.path.join(out_dir, f"{args.symbol}_{args.interval}_v9.keras"))
    
    # Plotting
    import matplotlib.pyplot as plt
    pred_log = model.predict(X_val, verbose=0)
    pred_p = b_val[:, None] * np.exp(pred_log)
    true_p = b_val[:, None] * np.exp(y_val)
    
    # H=1 and H=10
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(true_p[:, 0], label="True")
    plt.plot(pred_p[:, 0], label="Pred", alpha=0.8)
    plt.title(f"{args.symbol} {args.interval} Horizon=1")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(true_p[:, -1], label="True")
    plt.plot(pred_p[:, -1], label="Pred", alpha=0.8)
    plt.title(f"{args.symbol} {args.interval} Horizon={PRED_LEN}")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", "forecast.png"))
    print(f"Done. Saved to {out_dir}")

if __name__ == "__main__":
    main()
