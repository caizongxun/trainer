#!/usr/bin/env python3
"""colab_workflow_v12.py

V12 "Market Regime & Reversal" Trainer

Objective:
- Solve the "Trend vs Range" dilemma.
- Task 1: Predict Market Regime (0=Range/Chop, 1=Trend).
- Task 2: Predict Reversal Signal (0=None, 1=Buy, 2=Sell).
- The model learns that "Buy at Support" is valid in Range, but "Buy at Breakout" is valid in Trend.

New Features for Regime Detection:
- ADX (Trend Strength).
- Choppiness Index (Trend vs Chop).
- Bollinger Band Width (Volatility Squeeze).

Architecture:
- Shared Encoder -> [Regime Head] + [Reversal Head].
- Multi-task Loss: Loss = Reversal_Loss + 0.5 * Regime_Loss.

Run on Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v12.py | python3 - \
  --symbol BTCUSDT --interval 15m --epochs 60

Artifacts:
- model: ./all_models/models_v12/{symbol}/{symbol}_{interval}_v12_regime.keras
- plot : ./all_models/models_v12/{symbol}/plots/regime_signals.png
"""

import os
import argparse
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

def _configure_tf() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    except: pass

# ------------------------------
# Feature engineering (V12 Regime Aware)
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

def _choppiness(df, period=14):
    # 100 * LOG10( SUM(ATR(1), n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(period).sum()
    high_max = df['high'].rolling(period).max()
    low_min = df['low'].rolling(period).min()
    
    chop = 100 * np.log10(atr_sum / (high_max - low_min + 1e-12)) / np.log10(period)
    return chop.fillna(50)

def add_features_v12(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    
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

    # 1. Base Price Action
    d["log_ret"] = np.log(d["close"] / d["close"].shift(1))
    
    # 2. Volatility / Regime Indicators
    d["adx"] = _adx(d) / 100.0  # Normalize to 0-1
    d["chop"] = _choppiness(d) / 100.0 # Normalize to 0-1
    
    bb_mean = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_width"] = (4 * bb_std) / bb_mean
    d["bb_pos"] = (d["close"] - bb_mean) / (2 * bb_std + 1e-12)

    # 3. Momentum
    d["rsi"] = _wilder_rsi(d["close"]) / 100.0
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()

    # 4. Volume
    d["vol_z"] = (d["volume"] - d["volume"].rolling(20).mean()) / (d["volume"].rolling(20).std() + 1e-12)

    d = d.dropna().reset_index(drop=True)
    
    feature_cols = [
        "log_ret", "adx", "chop", "bb_width", "bb_pos",
        "rsi", "macd_hist", "vol_z"
    ]
    return d, feature_cols

# ------------------------------
# Targets: 
# 1. Reversal Label (0=None, 1=Buy, 2=Sell)
# 2. Regime Label (0=Range, 1=Trend)
# ------------------------------

def label_regime(df: pd.DataFrame, window=24):
    # Simple logic: High ADX (>25) = Trend(1), Low ADX = Range(0)
    # Or based on returns: if abs(return) > threshold consistently
    adx_raw = _adx(df)
    regime = np.zeros(len(df), dtype=int)
    regime[adx_raw > 25] = 1 # Trend
    return regime

def label_pivots(df: pd.DataFrame, left=12, right=12):
    # Strict pivot labeling
    highs = df["high"].values
    lows = df["low"].values
    labels = np.zeros(len(df), dtype=int)
    for i in range(left, len(df) - right):
        if highs[i] == np.max(highs[i-left : i+right+1]):
            labels[i] = 2 # Sell (Top)
        elif lows[i] == np.min(lows[i-left : i+right+1]):
            labels[i] = 1 # Buy (Bottom)
    return labels

def create_sequences_multitask(df: pd.DataFrame, cols: list[str], 
                               rev_labels: np.ndarray, reg_labels: np.ndarray, 
                               seq_len: int, horizon: int):
    feat = df[cols].values.astype(np.float32)
    max_i = len(df) - seq_len - horizon
    
    X = np.zeros((max_i, seq_len, len(cols)), dtype=np.float32)
    y_rev = np.zeros((max_i,), dtype=int)
    y_reg = np.zeros((max_i,), dtype=int)
    
    for i in range(max_i):
        X[i] = feat[i : i+seq_len]
        
        # Reversal Target: Is there a pivot in next horizon?
        fut_rev = rev_labels[i+seq_len : i+seq_len+horizon]
        if 2 in fut_rev: y_rev[i] = 2
        elif 1 in fut_rev: y_rev[i] = 1
        
        # Regime Target: What is the dominant regime in next horizon?
        fut_reg = reg_labels[i+seq_len : i+seq_len+horizon]
        y_reg[i] = 1 if np.mean(fut_reg) > 0.5 else 0
        
    return X, y_rev, y_reg, df["open_time"].values[seq_len+horizon-1 : seq_len+horizon-1+max_i]

# ------------------------------
# Model: Multi-task
# ------------------------------

def build_model_v12(seq_len: int, n_feat: int):
    from tensorflow.keras import layers, Model
    
    inp = layers.Input(shape=(seq_len, n_feat))
    
    # Shared Encoder
    x = layers.Conv1D(64, 3, activation="swish", padding="same")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.3))(x)
    
    # Head 1: Regime Classification (Binary)
    h1 = layers.Dense(32, activation="swish")(x)
    out_reg = layers.Dense(1, activation="sigmoid", name="regime")(h1)
    
    # Head 2: Reversal Classification (3-class)
    # We inject Regime info into Reversal head to help it decide
    h2 = layers.Concatenate()([x, out_reg]) 
    h2 = layers.Dense(64, activation="swish")(h2)
    out_rev = layers.Dense(3, activation="softmax", name="reversal")(h2)
    
    model = Model(inp, [out_rev, out_reg], name="v12_multitask")
    
    model.compile(
        optimizer="adam",
        loss={
            "reversal": "sparse_categorical_crossentropy",
            "regime": "binary_crossentropy"
        },
        loss_weights={"reversal": 1.0, "regime": 0.5},
        metrics={"reversal": "accuracy", "regime": "accuracy"}
    )
    return model

# ------------------------------
# Main
# ------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()

    _print_step("1/4", "Setup")
    _configure_tf()
    _set_seed(42)
    
    _print_step("2/4", "Data")
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
    df, features = add_features_v12(df)
    
    # Labels
    labels_rev = label_pivots(df, left=12, right=12) # Strict pivots
    labels_reg = label_regime(df)
    
    SEQ_LEN = 96
    HORIZON = 6
    
    train_size = int(len(df) * 0.9)
    scaler = StandardScaler()
    df.loc[:train_size-1, features] = scaler.fit_transform(df.loc[:train_size-1, features])
    df.loc[train_size:, features] = scaler.transform(df.loc[train_size:, features])
    
    X, y_rev, y_reg, t_arr = create_sequences_multitask(df, features, labels_rev, labels_reg, SEQ_LEN, HORIZON)
    
    split_idx = int(len(X) * 0.9)
    X_train, y_rev_train, y_reg_train = X[:split_idx], y_rev[:split_idx], y_reg[:split_idx]
    X_val, y_rev_val, y_reg_val = X[split_idx:], y_rev[split_idx:], y_reg[split_idx:]
    
    # Fix Keras multi-output class weight issue
    # Solution: Remove class_weight arg and handle imbalance by not forcing weights or by using sample_weight
    # For now, let's keep it simple and rely on Focal Loss if needed (but currently using sparse_categorical)
    # Or just let it learn naturally as we have Strict Labeling now which is cleaner.
    
    _print_step("3/4", "Train V12")
    model = build_model_v12(SEQ_LEN, len(features))
    
    cb = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")]
    
    # REMOVED class_weight argument to fix the ValueError
    model.fit(
        X_train, {"reversal": y_rev_train, "regime": y_reg_train},
        validation_data=(X_val, {"reversal": y_rev_val, "regime": y_reg_val}),
        epochs=args.epochs,
        batch_size=64,
        callbacks=cb,
        verbose=1
    )
    
    _print_step("4/4", "Plot")
    out_dir = f"./all_models/models_v12/{args.symbol}"
    _safe_mkdir(os.path.join(out_dir, "plots"))
    model.save(os.path.join(out_dir, f"{args.symbol}_{args.interval}_v12_regime.keras"))
    
    # Predict
    pred_rev, pred_reg = model.predict(X_val, verbose=0)
    rev_cls = np.argmax(pred_rev, axis=1) # 0,1,2
    reg_prob = pred_reg.flatten() # 0..1
    
    # Plot
    import matplotlib.pyplot as plt
    val_df = df.iloc[-len(y_rev_val):].copy().reset_index(drop=True)
    price = val_df["close"].values
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 1. Price + Signals
    ax1.plot(price, color="gray", alpha=0.5, label="Price")
    
    # Buy (Class 1)
    buy_idx = np.where(rev_cls == 1)[0]
    ax1.scatter(buy_idx, price[buy_idx], color="green", marker="^", s=40, label="Buy Signal")
    
    # Sell (Class 2)
    sell_idx = np.where(rev_cls == 2)[0]
    ax1.scatter(sell_idx, price[sell_idx], color="red", marker="v", s=40, label="Sell Signal")
    
    ax1.set_title(f"V12 Market Regime & Reversal: {args.symbol} {args.interval}")
    ax1.legend()
    
    # 2. Regime
    ax2.plot(reg_prob, color="purple", label="Trend Prob (ADX-based)")
    ax2.axhline(0.5, color="black", linestyle="--", alpha=0.3)
    ax2.fill_between(range(len(reg_prob)), reg_prob, 0.5, where=(reg_prob>0.5), color="purple", alpha=0.1, label="Trend Regime")
    ax2.fill_between(range(len(reg_prob)), reg_prob, 0.5, where=(reg_prob<=0.5), color="gray", alpha=0.1, label="Range Regime")
    ax2.set_ylabel("Regime (0=Range, 1=Trend)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", "regime_signals.png"))
    print(f"Done. Saved to {out_dir}")

if __name__ == "__main__":
    main()
