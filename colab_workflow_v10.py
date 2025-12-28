#!/usr/bin/env python3
"""colab_workflow_v10.py

V10 "Range Forecast" (Support/Resistance) Trainer

Objective
- Stop forecasting point price directly.
- Forecast a *tradable range* for the next N candles:
    - Upper bound (resistance proxy): q90 of future highs in the next horizon
    - Lower bound (support proxy): q10 of future lows in the next horizon

This reframes the problem from "predict the exact close" to "predict where price is likely bounded".
It is generally more actionable for entries/exits and reduces sensitivity to micro-noise.

Default:
- symbol: BTCUSDT
- interval: 15m
- seq_len: 96 (about 24h)
- horizon: 10 (about 2.5h)
- bounds: q10/q90

Run on Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v10.py | python3 - \
  --symbol BTCUSDT --interval 15m --epochs 60 --time_budget_min 120

Artifacts
- model: ./all_models/models_v10/{symbol}/{symbol}_{interval}_v10_range.keras
- plot : ./all_models/models_v10/{symbol}/plots/forecast_range.png
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
    except Exception:
        pass

    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass


# ------------------------------
# Feature engineering (reuse V9.5 set)
# ------------------------------

def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _parse_time_col(d: pd.DataFrame) -> pd.Series:
    time_col = next((c for c in ["open_time", "opentime", "timestamp", "time", "date"] if c in d.columns), None)
    if not time_col:
        raise ValueError("Missing time column")

    if pd.api.types.is_numeric_dtype(d[time_col]):
        ts = d[time_col].astype("int64")
        unit = "ms" if ts.median() > 10**12 else "s"
        return pd.to_datetime(ts, unit=unit, utc=True)

    return pd.to_datetime(d[time_col], utc=True, errors="coerce")


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]

    d["open_time"] = _parse_time_col(d)
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna().sort_values("open_time").reset_index(drop=True)

    # Base
    d["log_ret"] = np.log(d["close"] / d["close"].shift(1))
    d["upper_shadow"] = (d["high"] - np.maximum(d["close"], d["open"])) / d["close"]
    d["lower_shadow"] = (np.minimum(d["close"], d["open"]) - d["low"]) / d["close"]
    d["body"] = (d["close"] - d["open"]) / d["open"]
    d["rsi"] = _wilder_rsi(d["close"]) / 100.0

    # Volume
    d["log_vol"] = np.log1p(d["volume"])
    d["log_vol_diff"] = d["log_vol"].diff()

    # Time cyclical
    d["hour_sin"] = np.sin(2 * np.pi * d["open_time"].dt.hour / 24.0)
    d["dow_sin"] = np.sin(2 * np.pi * d["open_time"].dt.dayofweek / 7.0)

    # Trend distances
    for span in [7, 25, 99]:
        ema = d["close"].ewm(span=span, adjust=False).mean()
        d[f"dist_ema{span}"] = np.log(d["close"] / ema)

    # Momentum / Acceleration
    d["roc_6"] = d["close"].pct_change(6)
    d["roc_12"] = d["close"].pct_change(12)

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = macd - signal

    d = d.dropna().reset_index(drop=True)

    feature_cols = [
        "log_ret", "upper_shadow", "lower_shadow", "body",
        "rsi", "log_vol", "log_vol_diff", "hour_sin", "dow_sin",
        "dist_ema7", "dist_ema25", "dist_ema99",
        "roc_6", "roc_12", "macd_hist",
    ]

    return d, feature_cols


# ------------------------------
# Targets: q10/q90 of future low/high over the horizon
# We train on log-return vs base_close (end of input window)
# ------------------------------

def create_sequences_range(df: pd.DataFrame, cols: list[str], seq_len: int, horizon: int,
                           q_low: float, q_high: float):
    feat = df[cols].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)
    high = df["high"].values.astype(np.float32)
    low = df["low"].values.astype(np.float32)
    t = df["open_time"].values

    n = len(df)
    max_i = n - seq_len - horizon

    X = np.zeros((max_i, seq_len, len(cols)), dtype=np.float32)
    y = np.zeros((max_i, 2), dtype=np.float32)  # [upper_logret, lower_logret]
    base_close = np.zeros((max_i,), dtype=np.float32)
    end_time = np.empty((max_i,), dtype="datetime64[ns]")

    for i in range(max_i):
        X[i] = feat[i:i + seq_len]
        base_idx = i + seq_len - 1
        bc = close[base_idx]
        base_close[i] = bc

        fut_hi = high[i + seq_len:i + seq_len + horizon]
        fut_lo = low[i + seq_len:i + seq_len + horizon]

        upper = np.quantile(fut_hi, q_high)
        lower = np.quantile(fut_lo, q_low)

        # log-return of bounds vs base
        y[i, 0] = np.log(upper / bc)
        y[i, 1] = np.log(lower / bc)

        # align label to the end of the forecast window
        end_time[i] = t[i + seq_len + horizon - 1]

    return X, y, base_close, end_time


# ------------------------------
# Model
# Output in log-return space:
# - upper in (0, +cap)
# - lower in (-cap, 0)
# ------------------------------

def build_model_v10(seq_len: int, n_feat: int, cap: float = 0.15, lr: float = 1e-3):
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(seq_len, n_feat))

    x = layers.Conv1D(64, 3, activation="swish", padding="same")(inp)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.2))(x)

    x = layers.Dense(128, activation="swish")(x)
    x = layers.Dropout(0.2)(x)

    raw = layers.Dense(2)(x)
    raw_u = layers.Lambda(lambda z: z[:, :1])(raw)
    raw_l = layers.Lambda(lambda z: z[:, 1:])(raw)

    # enforce sign and cap
    upper = layers.Lambda(lambda z: cap * tf.nn.sigmoid(z))(raw_u)          # (0, cap)
    lower = layers.Lambda(lambda z: -cap * tf.nn.sigmoid(z))(raw_l)         # (-cap, 0)

    out = layers.Concatenate(axis=1)([upper, lower])

    model = Model(inp, out, name="v10_range_forecast")

    def pinball(y_true, y_pred, q: float):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))

    def loss_fn(y_true, y_pred):
        # y_true[:,0]=upper, y_true[:,1]=lower
        upper_loss = pinball(y_true[:, 0], y_pred[:, 0], 0.90)
        lower_loss = pinball(y_true[:, 1], y_pred[:, 1], 0.10)
        # small regularizer to avoid too-wide bands
        width = tf.reduce_mean(tf.square(y_pred[:, 0] - y_pred[:, 1]))
        return upper_loss + lower_loss + 0.05 * width

    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-3)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["mae"])
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--time_budget_min", type=int, default=120)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--cap", type=float, default=0.15)
    args = p.parse_args()

    _print_step("1/4", "Setup")
    _configure_tf()
    _set_seed(42)

    _print_step("2/4", "Data")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id="zongowo111/cpb-models",
        repo_type="dataset",
        allow_patterns=None,
        ignore_patterns=None,
    )

    csv_file = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv") and args.symbol in f and args.interval in f:
                csv_file = os.path.join(root, f)
                break
        if csv_file:
            break

    if not csv_file:
        raise ValueError(f"No CSV found for {args.symbol} {args.interval}")

    _print_kv("csv", csv_file)

    df = pd.read_csv(csv_file)
    df, features = add_features(df)

    # split
    train_size = int(len(df) * 0.9)

    # scale X only
    scaler = StandardScaler()
    df.loc[:train_size - 1, features] = scaler.fit_transform(df.loc[:train_size - 1, features])
    df.loc[train_size:, features] = scaler.transform(df.loc[train_size:, features])

    X, y, base_close, end_time = create_sequences_range(
        df, features, args.seq_len, args.horizon, q_low=0.10, q_high=0.90
    )

    # build mask by end_time boundary
    train_mask = np.arange(len(end_time)) < (train_size - args.seq_len - args.horizon)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[~train_mask], y[~train_mask]
    b_val = base_close[~train_mask]
    t_val = end_time[~train_mask]

    _print_kv("X_train", X_train.shape)
    _print_kv("X_val", X_val.shape)
    _print_kv("features", len(features))

    _print_step("3/4", "Train V10")
    model = build_model_v10(args.seq_len, len(features), cap=args.cap, lr=1e-3)

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss"),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=64,
        callbacks=cb,
        verbose=1,
    )

    _print_step("4/4", "Save & Plot")
    out_dir = f"./all_models/models_v10/{args.symbol}"
    _safe_mkdir(os.path.join(out_dir, "plots"))

    model_path = os.path.join(out_dir, f"{args.symbol}_{args.interval}_v10_range.keras")
    model.save(model_path)
    _print_kv("saved_model", model_path)

    # predict
    pred = model.predict(X_val, verbose=0)
    pred_upper = b_val * np.exp(pred[:, 0])
    pred_lower = b_val * np.exp(pred[:, 1])

    true_upper = b_val * np.exp(y_val[:, 0])
    true_lower = b_val * np.exp(y_val[:, 1])

    # Plot as time-aligned band (using end_time)
    import matplotlib.pyplot as plt

    # Downsample for readability
    n = min(len(t_val), 1200)
    t_plot = pd.to_datetime(t_val[:n])

    plt.figure(figsize=(14, 8))
    ax = plt.gca()

    ax.plot(t_plot, true_upper[:n], color="#1f77b4", linewidth=1.0, label="True Upper(q90)")
    ax.plot(t_plot, true_lower[:n], color="#1f77b4", linewidth=1.0, alpha=0.6, label="True Lower(q10)")

    ax.plot(t_plot, pred_upper[:n], color="#ff7f0e", linewidth=1.0, label="Pred Upper(q90)")
    ax.plot(t_plot, pred_lower[:n], color="#ff7f0e", linewidth=1.0, alpha=0.7, label="Pred Lower(q10)")

    ax.fill_between(t_plot, pred_lower[:n], pred_upper[:n], color="#ff7f0e", alpha=0.12, label="Pred Range")

    ax.set_title(f"{args.symbol} {args.interval} | Forecast Range for next {args.horizon} candles")
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.legend(loc="upper left")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "plots", "forecast_range.png")
    plt.savefig(fig_path)
    _print_kv("saved_plot", fig_path)


if __name__ == "__main__":
    main()
