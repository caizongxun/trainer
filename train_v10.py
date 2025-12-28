import os
import sys
import argparse
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import urllib.request
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, Attention, Dropout, Reshape, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber

# ============================================================
# v10 Trainer
# - Predict next PRED_LEN candles' (Close, High, Low)
# - Targets are log-returns relative to the last Close of input window
# - Features include log returns + technical indicators
# - RobustScaler for outlier-robust normalization
# - CNN + BiLSTM + Attention hybrid
# ============================================================

HF_DATASET_BASE = "https://huggingface.co/datasets/zongowo111/cpb-models/resolve/main"
HF_SUBDIR = "klines_binance_us"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_SEQ_LEN = 96
DEFAULT_PRED_LEN = 10
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_TRAIN_RATIO = 0.9
DEFAULT_SEED = 42


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _try_read_csv(url: str, timeout_sec: int = 30) -> pd.DataFrame | None:
    try:
        print(f"[DATA] Trying URL: {url}")
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            status = getattr(resp, "status", None)
            if status is not None:
                print(f"[DATA] HTTP status: {status}")
        df = pd.read_csv(url)
        print(f"[DATA] Download OK: rows={len(df)}")
        return df
    except Exception as e:
        print(f"[DATA] Download failed: {type(e).__name__}: {e}")
        return None


def download_data(symbol: str, interval: str) -> pd.DataFrame:
    """Download from HuggingFace. The dataset path in your summary json is:
    klines/{SYMBOL}/{SYMBOL}_{INTERVAL}_binance_us.csv

    The previous v10 script used a wrong path and caused 404.
    """
    candidates = [
        # Correct path based on your provided klines_summary_binance_us.json
        f"{HF_DATASET_BASE}/{HF_SUBDIR}/klines/{symbol}/{symbol}_{interval}_binance_us.csv",
        # Backward-compatible fallbacks (in case structure differs)
        f"{HF_DATASET_BASE}/{HF_SUBDIR}/{symbol}/{symbol}_{interval}.csv",
        f"{HF_DATASET_BASE}/{HF_SUBDIR}/{symbol}/{symbol}_{interval}_binance_us.csv",
    ]

    last_err = None
    df = None
    for url in candidates:
        df = _try_read_csv(url)
        if df is not None and len(df) > 0:
            break

    if df is None:
        raise RuntimeError("All dataset URL candidates failed. Please verify dataset path and access.")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Heuristic: find time column
    time_col = None
    for c in ["open_time", "opentime", "timestamp", "time", "datetime", "date"]:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        raise ValueError(f"Cannot find time column. Available columns: {list(df.columns)}")

    # Standardize OHLCV names for pandas_ta
    rename_map = {}
    for k, v in {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }.items():
        if k in df.columns:
            rename_map[k] = v

    df.rename(columns=rename_map, inplace=True)

    if not all(c in df.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
        raise ValueError(
            "Missing required OHLCV columns after rename. "
            f"Columns now: {list(df.columns)}"
        )

    # Parse datetime
    # If numeric and looks like ms since epoch
    if pd.api.types.is_numeric_dtype(df[time_col]):
        ts = df[time_col].astype("int64")
        # 1e12 ~ 2001-09-09 in ms. Most exchanges use ms.
        unit = "ms" if ts.median() > 10**12 else "s"
        df["open_time"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        df["open_time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    df.dropna(subset=["open_time"], inplace=True)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(
        "[DATA] Loaded columns: "
        + ",".join([c for c in df.columns if c != "open_time"])
    )
    print(f"[DATA] Time range: {df['open_time'].min()} -> {df['open_time'].max()}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[FE] Generating technical indicators...")

    # Trend
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.macd(append=True)  # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

    # Momentum
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(append=True)  # STOCHk_14_3_3, STOCHd_14_3_3

    # Volatility
    df.ta.atr(length=14, append=True)  # ATRr_14 or ATR_14 depending on version
    df.ta.bbands(length=20, std=2, append=True)  # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, ...

    # Volume-related
    df.ta.obv(append=True)

    # Stationarity-friendly core features
    df["log_ret_close"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_ret_open"] = np.log(df["Open"] / df["Open"].shift(1))
    df["log_ret_high"] = np.log(df["High"] / df["High"].shift(1))
    df["log_ret_low"] = np.log(df["Low"] / df["Low"].shift(1))
    df["log_vol"] = np.log(df["Volume"] + 1.0)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[FE] Data shape after features: {df.shape}")
    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    # Prefer explicit + stable columns (avoid accidental leakage)
    base = [
        "log_ret_close",
        "log_ret_open",
        "log_ret_high",
        "log_ret_low",
        "log_vol",
    ]

    # Add technical indicators if present
    optional_prefixes = [
        "EMA_",
        "MACD_",
        "MACDh_",
        "MACDs_",
        "RSI_",
        "STOCHk_",
        "STOCHd_",
        "ATR_",
        "ATRr_",
        "BBL_",
        "BBM_",
        "BBU_",
        "BBB_",
        "BBP_",
        "OBV",
    ]

    cols = []
    for c in df.columns:
        if c == "open_time":
            continue
        if c in ["Open", "High", "Low", "Close", "Volume"]:
            continue
        cols.append(c)

    chosen = []
    for c in base:
        if c in df.columns:
            chosen.append(c)

    for c in cols:
        for p in optional_prefixes:
            if c.startswith(p):
                chosen.append(c)
                break

    # De-duplicate while preserving order
    dedup = []
    seen = set()
    for c in chosen:
        if c not in seen:
            dedup.append(c)
            seen.add(c)

    if len(dedup) == 0:
        raise ValueError("No features selected. Please inspect feature engineering output.")

    print(f"[FE] Selected feature count: {len(dedup)}")
    print("[FE] Selected features: " + ",".join(dedup))
    return dedup


def create_sequences(
    feature_matrix: np.ndarray,
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:

    n = len(feature_matrix)
    max_i = n - seq_len - pred_len
    if max_i <= 0:
        raise ValueError(
            f"Not enough rows to create sequences. rows={n}, seq_len={seq_len}, pred_len={pred_len}"
        )

    X = np.zeros((max_i, seq_len, feature_matrix.shape[1]), dtype=np.float32)
    y = np.zeros((max_i, pred_len, 3), dtype=np.float32)

    print(f"[SEQ] Building sequences: total={max_i}, seq_len={seq_len}, pred_len={pred_len}")

    for i in range(max_i):
        X[i] = feature_matrix[i : i + seq_len]

        base_idx = i + seq_len - 1
        base_price = close_prices[base_idx]

        # Targets are future (close, high, low) log-returns relative to base_price
        for j in range(pred_len):
            future_idx = i + seq_len + j
            y[i, j, 0] = np.log(close_prices[future_idx] / base_price)
            y[i, j, 1] = np.log(high_prices[future_idx] / base_price)
            y[i, j, 2] = np.log(low_prices[future_idx] / base_price)

    return X, y


def build_v10_model(input_shape: tuple[int, int], pred_len: int, lr: float) -> Model:
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=64, kernel_size=3, activation="swish", padding="same")(inputs)
    x = Conv1D(filters=64, kernel_size=3, activation="swish", padding="same")(x)

    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(x)
    lstm_out = Dropout(0.2)(lstm_out)

    att_out = Attention()([lstm_out, lstm_out])
    x = Add()([lstm_out, att_out])

    x = Bidirectional(LSTM(64, return_sequences=False))(x)

    x = Dense(128, activation="swish")(x)
    x = Dropout(0.2)(x)

    out = Dense(pred_len * 3)(x)
    out = Reshape((pred_len, 3))(out)

    model = Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=Huber(delta=1.0),
        metrics=["mse", "mae"],
    )

    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    p.add_argument("--interval", type=str, default=DEFAULT_INTERVAL)
    p.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--pred_len", type=int, default=DEFAULT_PRED_LEN)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("[ENV] Python:", sys.version.replace("\n", " "))
    print("[ENV] TensorFlow:", tf.__version__)
    print("[ENV] Num GPUs:", len(tf.config.list_physical_devices("GPU")))
    
    print(
        "[CFG] symbol=", args.symbol,
        " interval=", args.interval,
        " seq_len=", args.seq_len,
        " pred_len=", args.pred_len,
        " epochs=", args.epochs,
        " batch_size=", args.batch_size,
        " lr=", args.lr,
        " train_ratio=", args.train_ratio,
        " seed=", args.seed,
    )

    # 1) Load + features
    df = download_data(args.symbol, args.interval)
    df = add_features(df)

    # 2) Split by time
    n = len(df)
    train_n = int(n * args.train_ratio)
    if train_n <= args.seq_len + args.pred_len:
        raise ValueError("Train split too small. Increase train_ratio or decrease seq_len/pred_len.")

    df_train = df.iloc[:train_n].copy()
    df_val = df.iloc[train_n:].copy()

    feature_cols = select_feature_columns(df_train)

    # 3) Fit scaler on train, apply to both
    scaler = RobustScaler()
    train_feat = scaler.fit_transform(df_train[feature_cols].values)
    val_feat = scaler.transform(df_val[feature_cols].values)

    # 4) Build sequences
    X_train, y_train = create_sequences(
        train_feat,
        df_train["Close"].values,
        df_train["High"].values,
        df_train["Low"].values,
        args.seq_len,
        args.pred_len,
    )

    X_val, y_val = create_sequences(
        val_feat,
        df_val["Close"].values,
        df_val["High"].values,
        df_val["Low"].values,
        args.seq_len,
        args.pred_len,
    )

    print(f"[DATA] X_train={X_train.shape} y_train={y_train.shape}")
    print(f"[DATA] X_val={X_val.shape} y_val={y_val.shape}")

    # 5) Model
    model = build_v10_model((args.seq_len, X_train.shape[-1]), args.pred_len, args.lr)
    model.summary()

    # 6) Training
    model_path = f"{args.symbol}_{args.interval}_v10.keras"

    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", mode="min"),
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("[DONE] Training complete. Best model saved to:", model_path)

    # 7) Eval
    eval_out = model.evaluate(X_val, y_val, verbose=1)
    print("[EVAL]", eval_out)


if __name__ == "__main__":
    main()
