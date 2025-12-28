import os
import sys
import argparse
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import tensorflow as tf
import urllib.request
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Huber

# Reuse functions from train_v10.py for consistency
# We assume train_v10.py is in the same directory or we duplicate the logic
# To be standalone, we'll duplicate the necessary processing logic here

HF_DATASET_BASE = "https://huggingface.co/datasets/zongowo111/cpb-models/resolve/main"
HF_SUBDIR = "klines_binance_us"

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
    candidates = [
        # Correct path based on klines_summary_binance_us.json
        f"{HF_DATASET_BASE}/{HF_SUBDIR}/klines/{symbol}/{symbol}_{interval}_binance_us.csv",
        # Backward-compatible fallbacks
        f"{HF_DATASET_BASE}/{HF_SUBDIR}/{symbol}/{symbol}_{interval}.csv",
        f"{HF_DATASET_BASE}/{HF_SUBDIR}/{symbol}/{symbol}_{interval}_binance_us.csv",
    ]

    df = None
    for url in candidates:
        df = _try_read_csv(url)
        if df is not None and len(df) > 0:
            break

    if df is None:
        raise RuntimeError("All dataset URL candidates failed. Please verify dataset path and access.")

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Find time col
    time_col = next((c for c in ["open_time", "opentime", "timestamp", "time"] if c in df.columns), None)
    if not time_col: raise ValueError("No time column found")
    
    # Rename OHLCV
    rename_map = {}
    for k, v in {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}.items():
        if k in df.columns: rename_map[k] = v
    df.rename(columns=rename_map, inplace=True)
    
    # Parse time
    if pd.api.types.is_numeric_dtype(df[time_col]):
        unit = "ms" if df[time_col].median() > 10**12 else "s"
        df["open_time"] = pd.to_datetime(df[time_col], unit=unit, utc=True)
    else:
        df["open_time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Must match train_v10.py EXACTLY
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.macd(append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.obv(append=True)

    df["log_ret_close"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_ret_open"] = np.log(df["Open"] / df["Open"].shift(1))
    df["log_ret_high"] = np.log(df["High"] / df["High"].shift(1))
    df["log_ret_low"] = np.log(df["Low"] / df["Low"].shift(1))
    df["log_vol"] = np.log(df["Volume"] + 1.0)
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def select_feature_columns(df: pd.DataFrame) -> list:
    # Logic from train_v10.py to select same features
    base = ["log_ret_close", "log_ret_open", "log_ret_high", "log_ret_low", "log_vol"]
    optional_prefixes = ["EMA_", "MACD_", "MACDh_", "MACDs_", "RSI_", "STOCHk_", "STOCHd_", "ATR_", "ATRr_", "BBL_", "BBM_", "BBU_", "BBB_", "BBP_", "OBV"]
    
    cols = []
    for c in df.columns:
        if c in ["open_time", "Open", "High", "Low", "Close", "Volume"]: continue
        cols.append(c)
        
    chosen = [c for c in base if c in df.columns]
    for c in cols:
        for p in optional_prefixes:
            if c.startswith(p):
                chosen.append(c)
                break
                
    return list(dict.fromkeys(chosen)) # dedup

def create_sequences(feat, close, seq_len):
    # Only need inputs (X) for visualization
    X = []
    indices = []
    for i in range(len(feat) - seq_len):
        X.append(feat[i : i+seq_len])
        indices.append(i + seq_len - 1) # The index of the 'base' candle (last input candle)
    return np.array(X), indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5, help="Number of random samples to visualize")
    args = parser.parse_args()

    # 1. Load Data
    df = download_data(args.symbol, args.interval)
    df = add_features(df)
    
    # 2. Prepare Features
    # Important: We must scale using the SAME logic as training. 
    # Since we don't have the saved scaler object, we fit on the first 90% (assuming training split)
    # to simulate the training environment, then transform the whole DF.
    # In production, you should save/load the scaler pickle.
    train_size = int(len(df) * 0.9)
    df_train = df.iloc[:train_size]
    
    feats = select_feature_columns(df)
    print(f"[INFO] Features: {len(feats)}")
    
    scaler = RobustScaler()
    scaler.fit(df_train[feats].values)
    
    # We want to visualize TEST data (unseen)
    df_test = df.iloc[train_size:].copy().reset_index(drop=True)
    X_test_scaled = scaler.transform(df_test[feats].values)
    
    # 3. Create Sequences
    X, base_indices = create_sequences(X_test_scaled, df_test["Close"].values, args.seq_len)
    
    # 4. Load Model
    # Need to provide custom_objects if using custom loss
    model = load_model(args.model_path, custom_objects={"Huber": Huber})
    print("[INFO] Model loaded successfully")
    
    # 5. Predict & Visualize
    # Pick random samples from test set
    if len(X) < args.samples:
        sample_idxs = range(len(X))
    else:
        # Avoid picking indices too close to the end where ground truth might be missing
        valid_range = len(X) - args.pred_len
        sample_idxs = np.random.choice(range(valid_range), args.samples, replace=False)
        sample_idxs.sort()

    plt.figure(figsize=(15, 5 * len(sample_idxs)))
    
    for i, idx in enumerate(sample_idxs):
        # Input sequence (last portion)
        # We want to show a bit of history + prediction + actual
        
        # Get prediction
        # Shape: (1, pred_len, 3) -> Close, High, Low log-returns
        pred_log_ret = model.predict(X[idx:idx+1], verbose=0)[0] 
        
        base_idx = base_indices[idx] # Index in df_test
        base_price = df_test.loc[base_idx, "Close"]
        base_time = df_test.loc[base_idx, "open_time"]
        
        # Reconstruct predicted prices
        # Pred = base * exp(log_ret)
        pred_close = base_price * np.exp(pred_log_ret[:, 0])
        pred_high = base_price * np.exp(pred_log_ret[:, 1])
        pred_low = base_price * np.exp(pred_log_ret[:, 2])
        
        # Ground Truth
        # We need next pred_len candles
        gt_start = base_idx + 1
        gt_end = base_idx + args.pred_len
        
        gt_dates = df_test.loc[gt_start:gt_end, "open_time"]
        gt_close = df_test.loc[gt_start:gt_end, "Close"]
        gt_high = df_test.loc[gt_start:gt_end, "High"]
        gt_low = df_test.loc[gt_start:gt_end, "Low"]
        
        # History (Context) - Show last 20 candles of input
        hist_len = 20
        hist_start = base_idx - hist_len + 1
        hist_dates = df_test.loc[hist_start:base_idx, "open_time"]
        hist_close = df_test.loc[hist_start:base_idx, "Close"]
        
        # Plotting
        ax = plt.subplot(len(sample_idxs), 1, i+1)
        
        # 1. History
        ax.plot(hist_dates, hist_close, label="History (Close)", color="gray", linestyle="--")
        
        # 2. Actual Future
        ax.plot(gt_dates, gt_close, label="Actual Close", color="green", linewidth=2)
        ax.fill_between(gt_dates, gt_low, gt_high, color='green', alpha=0.1, label="Actual Range (H-L)")
        
        # 3. Prediction
        # Helper to align dates (assuming 1h) - strictly we should use gt_dates if available
        # or generate dates if prediction goes beyond data
        pred_dates = gt_dates if len(gt_dates) == args.pred_len else pd.date_range(start=base_time + pd.Timedelta(args.interval), periods=args.pred_len, freq=args.interval)

        ax.plot(pred_dates, pred_close, label="Predicted Close", color="red", marker="o", markersize=4)
        ax.fill_between(pred_dates, pred_low, pred_high, color='red', alpha=0.1, label="Predicted Range (H-L)")
        
        ax.set_title(f"Sample #{idx} | Base Time: {base_time} | Base Price: {base_price:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = "prediction_results.png"
    plt.savefig(save_path)
    print(f"[DONE] Visualization saved to {save_path}")
    # Try to show in Colab
    try:
        from IPython.display import Image, display
        display(Image(filename=save_path))
    except:
        pass

if __name__ == "__main__":
    main()
