#!/usr/bin/env python3
"""colab_workflow_v9.py

V9 Multi-Horizon (30 -> 10) Crypto Kline Forecast Trainer

Key goals:
- Use last 30 candles to predict next 10 candles (multi-output / MIMO)
- Price-heavy feature set
- Auxiliary volatility head to learn volatility magnitude
- Reduce multi-step drift via:
  - direct multi-horizon prediction (no recursive feeding)
  - horizon-weighted loss (later steps weighted higher)
  - shape loss on first-differences across the forecast horizon

Workflow:
  1) env setup
  2) fetch data from HF dataset zongowo111/cpb-models (klines_binance_us)
  3) train (epochs up to 100 with early stopping + time budget)
  4) save to ./all_models/models_v9/
  5) (optional) upload the whole folder to HF (single upload_folder call)

Colab (GPU) one-liner:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v9.py | python3 -

Typical single-symbol tuning run (no upload):
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v9.py | python3 - \
  --symbol BTCUSDT --interval 15m --epochs 100 --time_budget_min 120 --upload 0

Notes:
- No emojis.
- HuggingFace token is read from:
  1) env HF_TOKEN
  2) CLI arg --hf_token
  3) interactive input()
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
from tensorflow.keras.callbacks import Callback


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


def _configure_tf(enable_xla: bool = True, enable_mixed_precision: bool = True) -> dict:
    info = {
        "tf_version": tf.__version__,
        "num_gpus": len(tf.config.list_physical_devices("GPU")),
        "mixed_precision": False,
        "xla": False,
    }

    # GPU memory growth
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    # Mixed precision
    if enable_mixed_precision:
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            info["mixed_precision"] = True
        except Exception:
            info["mixed_precision"] = False

    # XLA
    if enable_xla:
        try:
            tf.config.optimizer.set_jit(True)
            info["xla"] = True
        except Exception:
            info["xla"] = False

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
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def add_price_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]

    # Identify time column
    time_col = None
    for c in ["open_time", "opentime", "timestamp", "time", "datetime", "date"]:
        if c in d.columns:
            time_col = c
            break

    if time_col is None:
        raise ValueError(f"Cannot find time column in columns={list(d.columns)}")

    # Parse time
    if pd.api.types.is_numeric_dtype(d[time_col]):
        ts = d[time_col].astype("int64")
        unit = "ms" if ts.median() > 10**12 else "s"
        d["open_time"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        d["open_time"] = pd.to_datetime(d[time_col], utc=True, errors="coerce")

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in d.columns:
            raise ValueError(f"Missing required column '{c}'. columns={list(d.columns)}")
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["open_time", "open", "high", "low", "close", "volume"]).copy()
    d = d.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    # Core returns
    d["log_ret_close"] = np.log(d["close"] / d["close"].shift(1))
    d["log_ret_open"] = np.log(d["open"] / d["open"].shift(1))
    d["log_ret_high"] = np.log(d["high"] / d["high"].shift(1))
    d["log_ret_low"] = np.log(d["low"] / d["low"].shift(1))

    # Candle anatomy
    oc_mid = (d["open"] + d["close"]) / 2.0
    d["hl_range"] = (d["high"] - d["low"]) / (d["close"] + 1e-12)
    d["oc_range"] = (d["close"] - d["open"]) / (d["open"] + 1e-12)
    d["upper_wick"] = (d["high"] - np.maximum(d["open"], d["close"])) / (d["close"] + 1e-12)
    d["lower_wick"] = (np.minimum(d["open"], d["close"]) - d["low"]) / (d["close"] + 1e-12)
    d["mid_price"] = oc_mid / (d["close"] + 1e-12)

    # Volume transforms
    d["log_vol"] = np.log(d["volume"] + 1.0)
    d["log_vol_chg"] = d["log_vol"].diff()

    # Rolling stats on returns
    for w in [3, 5, 10]:
        d[f"ret_mean_{w}"] = d["log_ret_close"].rolling(w).mean()
        d[f"ret_std_{w}"] = d["log_ret_close"].rolling(w).std()

    # Trend (EMA ratios)
    for span in [5, 10, 20, 50]:
        d[f"ema_{span}"] = _ema(d["close"], span=span) / (d["close"] + 1e-12)

    # RSI
    d["rsi_14"] = _wilder_rsi(d["close"], period=14) / 100.0

    # ATR (relative)
    d["atr_14"] = _atr(d, period=14) / (d["close"] + 1e-12)

    # Bollinger (20)
    bb_mid = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    bb_up = bb_mid + 2.0 * bb_std
    bb_dn = bb_mid - 2.0 * bb_std
    d["bb_width"] = (bb_up - bb_dn) / (bb_mid + 1e-12)
    d["bb_pos"] = (d["close"] - bb_dn) / ((bb_up - bb_dn) + 1e-12)

    # Time features
    ot = d["open_time"]
    hour = ot.dt.hour.astype(np.float32)
    dow = ot.dt.dayofweek.astype(np.float32)
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    d["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    d["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    d = d.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    feature_cols = [
        "log_ret_close",
        "log_ret_open",
        "log_ret_high",
        "log_ret_low",
        "hl_range",
        "oc_range",
        "upper_wick",
        "lower_wick",
        "mid_price",
        "log_vol",
        "log_vol_chg",
        "ret_mean_3",
        "ret_std_3",
        "ret_mean_5",
        "ret_std_5",
        "ret_mean_10",
        "ret_std_10",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_50",
        "rsi_14",
        "atr_14",
        "bb_width",
        "bb_pos",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    feature_cols = [c for c in feature_cols if c in d.columns]
    if len(feature_cols) < 10:
        raise ValueError(f"Too few features generated: {len(feature_cols)}")

    return d, feature_cols


# ------------------------------
# Data pipeline
# ------------------------------

def create_multihorizon_sequences(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat = df_feat[feature_cols].values.astype(np.float32)
    close = df_feat["close"].values.astype(np.float32)
    high = df_feat["high"].values.astype(np.float32)
    low = df_feat["low"].values.astype(np.float32)

    n = len(df_feat)
    max_i = n - seq_len - pred_len
    if max_i <= 0:
        raise ValueError(f"Not enough rows: rows={n}, seq_len={seq_len}, pred_len={pred_len}")

    X = np.zeros((max_i, seq_len, feat.shape[1]), dtype=np.float32)
    y_price = np.zeros((max_i, pred_len, 3), dtype=np.float32)
    y_vol = np.zeros((max_i, pred_len, 1), dtype=np.float32)
    base_close = np.zeros((max_i,), dtype=np.float32)
    end_index = np.zeros((max_i,), dtype=np.int32)

    for i in range(max_i):
        X[i] = feat[i : i + seq_len]
        base_idx = i + seq_len - 1
        bc = close[base_idx]
        base_close[i] = bc

        for j in range(pred_len):
            fi = i + seq_len + j
            y_price[i, j, 0] = math.log(max(close[fi] / bc, 1e-12))
            y_price[i, j, 1] = math.log(max(high[fi] / bc, 1e-12))
            y_price[i, j, 2] = math.log(max(low[fi] / bc, 1e-12))

            rng = max(high[fi] - low[fi], 0.0)
            y_vol[i, j, 0] = math.log(max(rng / bc, 1e-12))

        end_index[i] = i + seq_len + pred_len - 1

    return X, y_price, y_vol, base_close, end_index


def chronological_split_by_end_index(
    X: np.ndarray,
    y_price: np.ndarray,
    y_vol: np.ndarray,
    base_close: np.ndarray,
    end_index: np.ndarray,
    train_end_row: int,
) -> dict:
    train_mask = end_index < train_end_row
    val_mask = ~train_mask

    out = {
        "X_train": X[train_mask],
        "y_price_train": y_price[train_mask],
        "y_vol_train": y_vol[train_mask],
        "base_close_train": base_close[train_mask],
        "X_val": X[val_mask],
        "y_price_val": y_price[val_mask],
        "y_vol_val": y_vol[val_mask],
        "base_close_val": base_close[val_mask],
    }

    if len(out["X_train"]) < 200 or len(out["X_val"]) < 50:
        raise ValueError(f"Split too small: train={len(out['X_train'])}, val={len(out['X_val'])}")

    return out


# ------------------------------
# Model
# ------------------------------

def build_v9_model(seq_len: int, n_features: int, pred_len: int, lr: float) -> "object":
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(seq_len, n_features), name="x")

    x = layers.Conv1D(64, 3, padding="same", activation="swish")(inputs)
    x = layers.Conv1D(64, 3, padding="same", activation="swish")(x)

    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.15))(x)

    att = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    x = layers.Add()([x, att])
    x = layers.LayerNormalization()(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.15))(x)

    x = layers.Dense(192, activation="swish")(x)
    x = layers.Dropout(0.2)(x)

    price = layers.Dense(pred_len * 3, dtype="float32", name="price_dense")(x)
    price = layers.Reshape((pred_len, 3), name="price")(price)

    vol = layers.Dense(pred_len * 1, dtype="float32", name="vol_dense")(x)
    vol = layers.Reshape((pred_len, 1), name="vol")(vol)

    model = Model(inputs=inputs, outputs={"price": price, "vol": vol}, name="v9_multihorizon")

    delta = 1.0
    horizon_w = tf.reshape(tf.linspace(1.0, float(pred_len), pred_len), (1, pred_len, 1))
    horizon_w = horizon_w / tf.reduce_mean(horizon_w)

    def price_loss(y_true, y_pred):
        err = y_true - y_pred
        abs_err = tf.abs(err)
        quad = tf.minimum(abs_err, delta)
        lin = abs_err - quad
        huber = 0.5 * tf.square(quad) + delta * lin
        huber_w = huber * horizon_w
        base = tf.reduce_mean(huber_w)

        dy_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        shape = tf.reduce_mean(tf.square(dy_true - dy_pred))

        return base + 0.15 * shape

    def vol_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss={"price": price_loss, "vol": vol_loss},
        loss_weights={"price": 1.0, "vol": 0.25},
        metrics={"price": ["mae"], "vol": ["mae"]},
    )

    return model


def _reconstruct_close_from_returns(base_close: np.ndarray, pred_logret_close: np.ndarray) -> np.ndarray:
    return base_close.reshape(-1, 1) * np.exp(pred_logret_close)


def _mape(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a), 1e-12)
    return float(np.mean(np.abs((a - b) / denom)))


class ValMAPECallback(Callback):
    """Compute validation MAPE on reconstructed close prices and inject into logs."""

    def __init__(
        self,
        X_val: np.ndarray,
        y_price_val: np.ndarray,
        base_close_val: np.ndarray,
        pred_len: int,
        max_batches: int = 20,
    ):
        # Do NOT call super().__init__() with no args if using legacy Keras or certain TF versions
        # Just manually set model to None (Keras will set it later via set_model)
        self.model = None
        self.X_val = X_val
        self.y_price_val = y_price_val
        self.base_close_val = base_close_val
        self.pred_len = pred_len
        self.max_batches = max_batches
        self.val_mape_close = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        n = len(self.X_val)
        if n <= 0:
            return

        take = min(n, self.max_batches * 256)
        Xs = self.X_val[-take:]
        ys = self.y_price_val[-take:]
        bs = self.base_close_val[-take:]

        pred = self.model.predict(Xs, verbose=0)
        pred_price = pred["price"]

        true_close = _reconstruct_close_from_returns(bs, ys[:, :, 0])
        pred_close = _reconstruct_close_from_returns(bs, pred_price[:, :, 0])

        mape_close = _mape(true_close, pred_close)
        self.val_mape_close = mape_close
        logs["val_mape_close"] = mape_close


class TimeBudgetCallback(Callback):
    def __init__(self, deadline_ts: float):
        self.model = None
        self.deadline_ts = deadline_ts

    def on_batch_end(self, batch, logs=None):
        if time.time() > self.deadline_ts:
            self.model.stop_training = True


def _as_tf_dataset(X: np.ndarray, y_price: np.ndarray, y_vol: np.ndarray, batch_size: int, shuffle: bool) -> "object":
    ds = tf.data.Dataset.from_tensor_slices((X, {"price": y_price, "vol": y_vol}))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------------
# HF dataset operations
# ------------------------------

def hf_snapshot_download_klines(dataset_id: str, local_dir: str) -> list[str]:
    from huggingface_hub import snapshot_download

    allow = [
        "klines_binance_us/**/*.csv",
        "klines_binance_us/**/*.json",
    ]
    _print_kv("allow_patterns", allow)

    path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow,
    )

    csvs = []
    for root, _, files in os.walk(path):
        for fn in files:
            if fn.lower().endswith(".csv") and "klines_binance_us" in root.replace("\\", "/"):
                csvs.append(os.path.join(root, fn))

    csvs.sort()
    return csvs


def parse_symbol_interval_from_filename(filename: str) -> tuple[str, str] | None:
    base = os.path.basename(filename)
    m = re.match(r"^(?P<sym>[A-Z0-9]+)_(?P<intv>\d+[mhdw])", base)
    if m:
        return m.group("sym"), m.group("intv")

    parts = base.split("_")
    if len(parts) >= 2:
        sym = parts[0]
        intv = parts[1]
        if sym and intv:
            return sym, intv
    return None


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for a in ["open", "high", "low", "close", "volume"]:
        if a not in df.columns and a.capitalize() in df.columns:
            rename_map[a.capitalize()] = a
    df = df.rename(columns=rename_map)

    return df


def upload_models_folder_to_hf(dataset_id: str, local_models_dir: str, repo_subdir: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_folder(
        repo_id=dataset_id,
        repo_type="dataset",
        folder_path=local_models_dir,
        path_in_repo=repo_subdir,
        commit_message=f"Upload {repo_subdir} ({datetime.now(timezone.utc).isoformat()})",
    )


def _save_forecast_plots(
    out_dir: str,
    symbol: str,
    interval: str,
    base_close_val: np.ndarray,
    y_price_val: np.ndarray,
    pred_price: np.ndarray,
    pred_len: int,
) -> dict:
    """Save plots to out_dir/plots and return dict of paths."""

    import matplotlib.pyplot as plt

    plots_dir = os.path.join(out_dir, "plots")
    _safe_mkdir(plots_dir)

    true_close_all = _reconstruct_close_from_returns(base_close_val, y_price_val[:, :, 0])
    pred_close_all = _reconstruct_close_from_returns(base_close_val, pred_price[:, :, 0])

    # Plot A: single example trajectory (last val sample)
    i = -1
    steps = np.arange(1, pred_len + 1)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(steps, true_close_all[i], label="true_close")
    plt.plot(steps, pred_close_all[i], label="pred_close")
    plt.title(f"{symbol} {interval} | 10-step trajectory (one val sample)")
    plt.xlabel("horizon step")
    plt.ylabel("price")
    plt.legend()
    p1 = os.path.join(plots_dir, f"{symbol}_{interval}_v9_forecast_example.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close(fig)

    # Plot B: series comparison for horizon 1 and horizon pred_len across val set
    h1 = 0
    hN = pred_len - 1
    t = np.arange(len(true_close_all))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, true_close_all[:, h1], label="true_h1")
    axes[0].plot(t, pred_close_all[:, h1], label="pred_h1")
    axes[0].set_title(f"{symbol} {interval} | horizon=1 (next candle close)")
    axes[0].legend()

    axes[1].plot(t, true_close_all[:, hN], label=f"true_h{pred_len}")
    axes[1].plot(t, pred_close_all[:, hN], label=f"pred_h{pred_len}")
    axes[1].set_title(f"{symbol} {interval} | horizon={pred_len}")
    axes[1].legend()

    axes[1].set_xlabel("validation sample index (chronological)")
    p2 = os.path.join(plots_dir, f"{symbol}_{interval}_v9_forecast_series_h1_h{pred_len}.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close(fig)

    npz_path = os.path.join(plots_dir, f"{symbol}_{interval}_v9_val_pred_true_close.npz")
    np.savez_compressed(
        npz_path,
        true_close=true_close_all.astype(np.float32),
        pred_close=pred_close_all.astype(np.float32),
    )

    return {"plot_example": p1, "plot_series": p2, "npz": npz_path}


# ------------------------------
# Main workflow
# ------------------------------

@dataclass
class TrainConfig:
    dataset_id: str = "zongowo111/cpb-models"
    seq_len: int = 30
    pred_len: int = 10
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    train_ratio: float = 0.9
    seed: int = 42
    time_budget_min: int = 120
    max_models: int = 0
    intervals: tuple[str, ...] = ("15m", "1h")
    symbol: str = ""
    interval: str = ""
    hf_token: str = ""
    upload: int = 0


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id", type=str, default="zongowo111/cpb-models")
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--pred_len", type=int, default=10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time_budget_min", type=int, default=120)
    p.add_argument("--max_models", type=int, default=0)
    p.add_argument("--intervals", type=str, default="15m,1h")
    p.add_argument("--symbol", type=str, default="")
    p.add_argument("--interval", type=str, default="")
    p.add_argument("--hf_token", type=str, default="")
    p.add_argument("--upload", type=int, default=0)

    a = p.parse_args()
    intervals = tuple([s.strip() for s in a.intervals.split(",") if s.strip()])

    return TrainConfig(
        dataset_id=a.dataset_id,
        seq_len=a.seq_len,
        pred_len=a.pred_len,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        train_ratio=a.train_ratio,
        seed=a.seed,
        time_budget_min=a.time_budget_min,
        max_models=a.max_models,
        intervals=intervals,
        symbol=(a.symbol or "").strip().upper(),
        interval=(a.interval or "").strip(),
        hf_token=a.hf_token or "",
        upload=int(a.upload),
    )


def main() -> int:
    cfg = parse_args()

    start_ts = time.time()
    deadline_ts = start_ts + cfg.time_budget_min * 60

    _print_step("1/7", "Environment setup")

    try:
        from google.colab import drive  # noqa: F401
        _print_kv("colab", True)
    except Exception:
        _print_kv("colab", False)

    try:
        import tensorflow as tf

        _set_seed(cfg.seed)
        tf_info = _configure_tf(enable_xla=True, enable_mixed_precision=True)
        for k, v in tf_info.items():
            _print_kv(k, v)
    except Exception as e:
        print(f"TensorFlow import/config failed: {type(e).__name__}: {e}")
        return 1

    _print_step("2/7", "Dependencies check")
    try:
        import huggingface_hub  # noqa: F401
        _print_kv("huggingface_hub", "ok")
    except Exception:
        _print_kv("huggingface_hub", "missing (pip install huggingface-hub)")

    _print_step("3/7", "Fetch data from HuggingFace dataset")

    data_root = "./data"
    local_models_root = "./all_models"
    local_models_v9 = os.path.join(local_models_root, "models_v9")

    _safe_mkdir(data_root)
    _safe_mkdir(local_models_v9)

    try:
        csv_paths = hf_snapshot_download_klines(cfg.dataset_id, local_dir=data_root)
    except Exception as e:
        print(f"Dataset download failed: {type(e).__name__}: {e}")
        return 1

    pairs = []
    for pth in csv_paths:
        meta = parse_symbol_interval_from_filename(pth)
        if meta is None:
            continue
        sym, intv = meta
        if intv not in cfg.intervals:
            continue
        if cfg.symbol and sym != cfg.symbol:
            continue
        if cfg.interval and intv != cfg.interval:
            continue
        pairs.append((sym, intv, pth))

    pairs.sort(key=lambda x: (x[0], x[1]))

    if cfg.max_models and cfg.max_models > 0:
        pairs = pairs[: cfg.max_models]

    _print_kv("csv_files_found", len(csv_paths))
    _print_kv("pairs_to_train", len(pairs))
    _print_kv("intervals", cfg.intervals)
    if cfg.symbol:
        _print_kv("symbol", cfg.symbol)
    if cfg.interval:
        _print_kv("interval", cfg.interval)

    if not pairs:
        print("No training pairs found. Please verify dataset structure and filters.")
        return 1

    _print_step("4/7", "Training")

    from sklearn.preprocessing import RobustScaler
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    training_results = []

    for idx, (symbol, interval, csv_path) in enumerate(pairs, 1):
        if time.time() > deadline_ts - 120:
            print("Time budget nearly exhausted. Stop starting new models.")
            break

        model_start = time.time()
        print(f"\n[MODEL {idx}/{len(pairs)}] {symbol} {interval}")
        _print_kv("csv", csv_path)

        try:
            df_raw = load_csv(csv_path)
            df_feat, feature_cols = add_price_features(df_raw)

            n_rows = len(df_feat)
            train_end_row = int(n_rows * cfg.train_ratio)
            if train_end_row <= 50:
                raise ValueError(f"Train split too small: train_end_row={train_end_row}, rows={n_rows}")

            df_feat.loc[:, feature_cols] = df_feat[feature_cols].astype(np.float32)

            scaler = RobustScaler()
            scaler.fit(df_feat.loc[: train_end_row - 1, feature_cols].values)
            df_feat.loc[:, feature_cols] = scaler.transform(df_feat[feature_cols].values).astype(np.float32)

            X, y_price, y_vol, base_close, end_index = create_multihorizon_sequences(
                df_feat,
                feature_cols,
                seq_len=cfg.seq_len,
                pred_len=cfg.pred_len,
            )

            splits = chronological_split_by_end_index(
                X,
                y_price,
                y_vol,
                base_close,
                end_index,
                train_end_row=train_end_row,
            )

            X_train = splits["X_train"]
            y_price_train = splits["y_price_train"]
            y_vol_train = splits["y_vol_train"]
            X_val = splits["X_val"]
            y_price_val = splits["y_price_val"]
            y_vol_val = splits["y_vol_val"]
            base_close_val = splits["base_close_val"]

            _print_kv("X_train", X_train.shape)
            _print_kv("X_val", X_val.shape)
            _print_kv("features", len(feature_cols))

            ds_train = _as_tf_dataset(X_train, y_price_train, y_vol_train, cfg.batch_size, shuffle=True)
            ds_val = _as_tf_dataset(X_val, y_price_val, y_vol_val, cfg.batch_size, shuffle=False)

            model = build_v9_model(cfg.seq_len, X_train.shape[-1], cfg.pred_len, cfg.lr)

            out_dir = os.path.join(local_models_v9, symbol)
            _safe_mkdir(out_dir)
            ckpt_path = os.path.join(out_dir, f"{symbol}_{interval}_v9_best.keras")
            final_path = os.path.join(out_dir, f"{symbol}_{interval}_v9.keras")
            scaler_path = os.path.join(out_dir, f"{symbol}_{interval}_v9_scaler.json")
            meta_path = os.path.join(out_dir, f"{symbol}_{interval}_v9_meta.json")

            val_mape_cb = ValMAPECallback(X_val, y_price_val, base_close_val, cfg.pred_len)
            time_cb = TimeBudgetCallback(deadline_ts)
            callbacks = [
                ModelCheckpoint(ckpt_path, monitor="val_loss", mode="min", save_best_only=True),
                EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
                val_mape_cb,
                time_cb,
            ]

            hist = model.fit(
                ds_train,
                validation_data=ds_val,
                epochs=cfg.epochs,
                verbose=1,
                callbacks=callbacks,
            )

            model.save(final_path)

            scaler_payload = {
                "type": "RobustScaler",
                "center_": scaler.center_.tolist() if hasattr(scaler, "center_") else None,
                "scale_": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
                "feature_cols": feature_cols,
                "seq_len": cfg.seq_len,
                "pred_len": cfg.pred_len,
            }
            with open(scaler_path, "w", encoding="utf-8") as f:
                json.dump(scaler_payload, f, ensure_ascii=False, indent=2)

            pred = model.predict(X_val, verbose=0)
            pred_close = _reconstruct_close_from_returns(base_close_val, pred["price"][:, :, 0])
            true_close = _reconstruct_close_from_returns(base_close_val, y_price_val[:, :, 0])
            val_mape_close = _mape(true_close, pred_close)

            plot_paths = _save_forecast_plots(
                out_dir=out_dir,
                symbol=symbol,
                interval=interval,
                base_close_val=base_close_val,
                y_price_val=y_price_val,
                pred_price=pred["price"],
                pred_len=cfg.pred_len,
            )

            meta_payload = {
                "symbol": symbol,
                "interval": interval,
                "version": "v9",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rows": int(len(df_raw)),
                "rows_after_features": int(len(df_feat)),
                "n_features": int(len(feature_cols)),
                "seq_len": int(cfg.seq_len),
                "pred_len": int(cfg.pred_len),
                "epochs_requested": int(cfg.epochs),
                "epochs_ran": int(len(hist.history.get("loss", []))),
                "val_mape_close": float(val_mape_close),
                "train_seconds": float(time.time() - model_start),
                "final_model": os.path.basename(final_path),
                "best_checkpoint": os.path.basename(ckpt_path),
                "plots": plot_paths,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, ensure_ascii=False, indent=2)

            print("Result")
            _print_kv("val_mape_close", f"{val_mape_close:.6f}")
            _print_kv("saved", final_path)
            _print_kv("plot_example", plot_paths["plot_example"])
            _print_kv("plot_series", plot_paths["plot_series"])

            training_results.append(meta_payload)

        except Exception as e:
            print(f"Training failed for {symbol} {interval}: {type(e).__name__}: {e}")

        finally:
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()

    _print_step("5/7", "Save summary")

    summary = {
        "started_at": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "time_budget_min": cfg.time_budget_min,
        "dataset_id": cfg.dataset_id,
        "version": "v9",
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "intervals": list(cfg.intervals),
        "trained_models": len(training_results),
        "results": training_results,
    }

    _safe_mkdir(local_models_root)
    with open(os.path.join(local_models_root, "training_summary_v9.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _print_kv("summary_path", os.path.join(local_models_root, "training_summary_v9.json"))

    _print_step("6/7", "Upload models_v9 to HuggingFace (optional)")

    if cfg.upload != 1:
        print("Upload disabled (use --upload 1 to enable).")
    else:
        print("If upload is not needed now, just press Enter when prompted and the script will skip upload.")
        try:
            token = (cfg.hf_token or "").strip()
            if not token:
                token = (os.environ.get("HF_TOKEN", "") or "").strip()
            if not token:
                try:
                    token = input("Enter HuggingFace token (leave blank to skip upload): ").strip()
                except EOFError:
                    token = ""

            if token:
                upload_models_folder_to_hf(
                    dataset_id=cfg.dataset_id,
                    local_models_dir=local_models_v9,
                    repo_subdir="models_v9",
                    token=token,
                )
                print("Upload completed.")
            else:
                print("Upload skipped.")
        except Exception as e:
            print(f"Upload failed: {type(e).__name__}: {e}")

    _print_step("7/7", "Done")
    _print_kv("total_seconds", f"{time.time() - start_ts:.1f}")
    _print_kv("models_trained", len(training_results))
    print("Local models directory: ./all_models/models_v9")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
