#!/usr/bin/env python3
"""colab_workflow_v11.py

V11 "Reversal Hunter" Trainer

Objective:
- Stop predicting continuous price.
- Predict PROBABILITY of a "Pivot High" or "Pivot Low" occurring within the next horizon.
- This acts as a signal generator for entries (Buy on high prob of Pivot Low, Sell on Pivot High).

Target Logic (ZigZag-like):
- A candle is a Pivot Low if it is the lowest point in a local window [t-L, t+R].
- A candle is a Pivot High if it is the highest point in a local window [t-L, t+R].
- We train a Classifier to output probability: [Prob_NoPivot, Prob_PivotHigh, Prob_PivotLow].

Features:
- Classic price/volume features.
- Divergence proxies (RSI slope vs Price slope).
- Volume Climax indicators (z-score of volume).
- Bollinger Band proximity.

Run on Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v11.py | python3 - \
  --symbol BTCUSDT --interval 15m --epochs 60 --class_weight 1

Artifacts:
- model: ./all_models/models_v11/{symbol}/{symbol}_{interval}_v11_reversal.keras
- plot : ./all_models/models_v11/{symbol}/plots/reversal_signals.png
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
# Feature engineering (V11 Enhanced)
# ------------------------------

def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def add_features_v11(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    
    # Parse time
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
    d["body"] = (d["close"] - d["open"]) / d["open"]
    d["upper_shadow"] = (d["high"] - np.maximum(d["close"], d["open"])) / d["close"]
    d["lower_shadow"] = (np.minimum(d["close"], d["open"]) - d["low"]) / d["close"]

    # 2. Volume Climax (Z-score of volume)
    vol_mean = d["volume"].rolling(20).mean()
    vol_std = d["volume"].rolling(20).std()
    d["vol_z"] = (d["volume"] - vol_mean) / (vol_std + 1e-12)
    
    # 3. RSI & Divergence proxies
    d["rsi"] = _wilder_rsi(d["close"]) / 100.0
    # Slope of price vs slope of RSI (simple divergence proxy)
    d["price_slope"] = d["close"].diff(5) / d["close"].shift(5)
    d["rsi_slope"] = d["rsi"].diff(5)
    
    # 4. Bollinger Band Position (Reversal often happens at bands)
    bb_mean = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_pos"] = (d["close"] - bb_mean) / (2 * bb_std + 1e-12) # >1 means above upper band

    # 5. MACD Histogram (Momentum wane)
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = macd - signal

    d = d.dropna().reset_index(drop=True)
    
    feature_cols = [
        "log_ret", "body", "upper_shadow", "lower_shadow",
        "vol_z", "rsi", "price_slope", "rsi_slope",
        "bb_pos", "macd_hist"
    ]
    return d, feature_cols

# ------------------------------
# Target: Pivot labeling
# ------------------------------

def label_pivots(df: pd.DataFrame, left: int=5, right: int=5):
    # 0: None, 1: Pivot High, 2: Pivot Low
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    labels = np.zeros(n, dtype=int)
    
    for i in range(left, n - right):
        window_highs = highs[i-left : i+right+1]
        window_lows = lows[i-left : i+right+1]
        
        # Check Pivot High (Highest in window)
        if highs[i] == np.max(window_highs):
            labels[i] = 1
        # Check Pivot Low (Lowest in window)
        elif lows[i] == np.min(window_lows):
            labels[i] = 2
            
    return labels

def create_sequences_cls(df: pd.DataFrame, cols: list[str], labels: np.ndarray, seq_len: int, pred_horizon: int):
    # We want to predict if a Pivot will occur in the *next* `pred_horizon` steps.
    # Actually, simpler: Predict if the *next candle* (or near future) is the start of a reversal?
    # Let's try: Predict if any candle in [t+1, t+pred_horizon] is a Pivot High or Low.
    
    feat = df[cols].values.astype(np.float32)
    n = len(df)
    max_i = n - seq_len - pred_horizon
    
    X = np.zeros((max_i, seq_len, len(cols)), dtype=np.float32)
    y = np.zeros((max_i,), dtype=int) 
    
    # y encoding: 
    # 0 = No pivot in near future
    # 1 = Pivot High incoming
    # 2 = Pivot Low incoming
    # Note: If both happen, prioritize the first one or stronger one. Simple logic: first one.
    
    for i in range(max_i):
        X[i] = feat[i : i+seq_len]
        
        future_labels = labels[i+seq_len : i+seq_len+pred_horizon]
        
        has_high = 1 in future_labels
        has_low = 2 in future_labels
        
        if has_high and not has_low:
            y[i] = 1
        elif has_low and not has_high:
            y[i] = 2
        elif has_high and has_low:
            # Find which comes first
            first_high = np.where(future_labels == 1)[0][0]
            first_low = np.where(future_labels == 2)[0][0]
            y[i] = 1 if first_high < first_low else 2
        else:
            y[i] = 0
            
    return X, y, df["open_time"].values[seq_len+pred_horizon-1 : seq_len+pred_horizon-1+max_i]

# ------------------------------
# Model: 3-class Classifier
# ------------------------------

def build_model_v11(seq_len: int, n_feat: int, lr: float=1e-3):
    from tensorflow.keras import layers, Model
    
    inp = layers.Input(shape=(seq_len, n_feat))
    
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.3))(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    # Output: 3 classes (None, High, Low)
    out = layers.Dense(3, activation="softmax")(x)
    
    model = Model(inp, out, name="v11_reversal_hunter")
    
    opt = tf.keras.optimizers.AdamW(learning_rate=lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ------------------------------
# Main
# ------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seq_len", type=int, default=60) # Lookback
    p.add_argument("--horizon", type=int, default=6)  # Lookahead for pivot (1.5h)
    p.add_argument("--class_weight", type=int, default=1) 
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
    df, features = add_features_v11(df)
    
    # Labeling Pivots (ZigZag logic)
    # pivot window: left=5, right=5 means a local extrema over 11 candles
    raw_labels = label_pivots(df, left=8, right=8) 
    
    # Create dataset
    train_size = int(len(df) * 0.9)
    scaler = StandardScaler()
    df.loc[:train_size-1, features] = scaler.fit_transform(df.loc[:train_size-1, features])
    df.loc[train_size:, features] = scaler.transform(df.loc[train_size:, features])
    
    X, y, t_arr = create_sequences_cls(df, features, raw_labels, args.seq_len, args.horizon)
    
    # Split
    # We must cut based on time, not random shuffle
    split_idx = int(len(X) * 0.9)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    t_val = t_arr[split_idx:]
    
    # Class weights? Pivots are rare.
    # 0: ~90%, 1: ~5%, 2: ~5%
    # Lets compute simple weights
    from sklearn.utils import class_weight
    if args.class_weight:
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cw_dict = dict(enumerate(cw))
        print(f"Class weights: {cw_dict}")
    else:
        cw_dict = None

    _print_step("3/4", "Train V11")
    model = build_model_v11(args.seq_len, len(features), lr=1e-3)
    
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=64,
        class_weight=cw_dict,
        callbacks=cb,
        verbose=1
    )
    
    _print_step("4/4", "Save & Plot")
    out_dir = f"./all_models/models_v11/{args.symbol}"
    _safe_mkdir(os.path.join(out_dir, "plots"))
    model.save(os.path.join(out_dir, f"{args.symbol}_{args.interval}_v11_reversal.keras"))
    
    # Visualize Signals
    probs = model.predict(X_val, verbose=0) # (N, 3)
    pred_cls = np.argmax(probs, axis=1)
    
    # Plot last 500 candles of validation
    import matplotlib.pyplot as plt
    
    # recover price for plotting (we need close price corresponding to t_val)
    # t_val contains datetime, let's map back to df
    # naive approach: just take the tail of df that matches validation length
    val_df = df.iloc[-len(y_val):].copy().reset_index(drop=True)
    price = val_df["close"].values
    
    plt.figure(figsize=(14, 6))
    plt.plot(price, color="gray", alpha=0.5, label="Price")
    
    # Plot Buy Signals (Pred Class 2 = Pivot Low incoming)
    buy_idx = np.where(pred_cls == 2)[0]
    if len(buy_idx) > 0:
        plt.scatter(buy_idx, price[buy_idx], color="green", marker="^", s=30, label="Pred Buy (Pivot Low)")
        
    # Plot Sell Signals (Pred Class 1 = Pivot High incoming)
    sell_idx = np.where(pred_cls == 1)[0]
    if len(sell_idx) > 0:
        plt.scatter(sell_idx, price[sell_idx], color="red", marker="v", s=30, label="Pred Sell (Pivot High)")

    # Plot True Pivots for comparison (optional, but good for debug)
    # We need to re-extract true labels for this segment
    # y_val is the ground truth
    true_buys = np.where(y_val == 2)[0]
    true_sells = np.where(y_val == 1)[0]
    
    plt.scatter(true_buys, price[true_buys], color="lime", marker="x", s=15, alpha=0.5, label="True Low")
    plt.scatter(true_sells, price[true_sells], color="magenta", marker="x", s=15, alpha=0.5, label="True High")

    plt.title(f"V11 Reversal Hunter: {args.symbol} {args.interval}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", "reversal_signals.png"))
    print(f"Done. Saved to {out_dir}")

if __name__ == "__main__":
    main()
