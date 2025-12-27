#!/usr/bin/env python3
"""
Colab優化的進階虛擬貨幣價格預測模型訓練
- 自動GPU优化
- 寶體進度步旘迷功箱
- 陸稜上傳功能
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

# ======================== 第一步：Colab環境棄保 ========================
print("[1/7] Colab環境設定...")

try:
    from google.colab import drive
    IS_COLAB = True
    print("  ✔ Google Colab環境偵測成功")
except ImportError:
    IS_COLAB = False
    print("  ⚠ 本地環境模式")

# GPU模組配置
print("  GPU优化配置...")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  ✔ 偵測到 {len(gpus)} 個GPU")
    else:
        print("  ⚠ 未偵測到GPU")
except Exception as e:
    print(f"  ⚠ GPU配置警告: {e}")

# ======================== 第二步：安裝依賴 ========================
print("\n[2/7] 安裝依賴套件...")

packages = {
    'tensorflow': 'TensorFlow',
    'keras': 'Keras',
    'huggingface_hub': 'Hugging Face Hub',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'sklearn': 'Scikit-Learn',
    'requests': 'Requests'
}

for module, name in packages.items():
    try:
        __import__(module)
        print(f"  ✔ {name} 已安裝")
    except ImportError:
        print(f"  安裝 {name}...")
        os.system(f'pip install -q {module.replace("_", "-")}')

# ======================== 第三步：技術指標 ========================
print("\n[3/7] 技術指標計算...")

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = pd.Series(prices).ewm(span=fast).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow).mean().values
    macd = ema_fast - ema_slow
    signal_line = pd.Series(macd).ewm(span=signal).mean().values
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, num_std=2):
    sma = pd.Series(prices).rolling(window=period).mean().values
    std = pd.Series(prices).rolling(window=period).std().values
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def add_technical_indicators(df):
    close_prices = df['close'].values
    df['rsi'] = calculate_rsi(close_prices, period=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(close_prices)
    upper_bb, mid_bb, lower_bb = calculate_bollinger_bands(close_prices)
    df['bb_position'] = np.where(
        upper_bb == lower_bb,
        0.5,
        (close_prices - lower_bb) / (upper_bb - lower_bb)
    )
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

print("  ✔ 技術指標凖讀成功")

# ======================== 第四步：下載資料 ========================
print("\n[4/7] 資料下載扣調...")

from huggingface_hub import hf_hub_download
from sklearn.preprocessing import MinMaxScaler

dataset_name = "zongowo111/cpb-models"
repo_type = "dataset"
os.makedirs("./data", exist_ok=True)
os.makedirs("./all_models", exist_ok=True)

try:
    summary_path = hf_hub_download(
        repo_id=dataset_name,
        filename="klines_binance_us/klines_summary_binance_us.json",
        repo_type=repo_type,
        local_dir="./data"
    )
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"  ✔ 找到 {len(summary)} 個幣種")
except Exception as e:
    print(f"  ⚠ 資料下載失敗: {e}")
    sys.exit(1)

# 下載胃數個幣種
download_limit = 10  # 可修改此數字

downloaded_pairs = []
for idx, (symbol, timeframes) in enumerate(list(summary.items())[:download_limit]):
    for timeframe in timeframes:
        filename = f"klines_binance_us/{symbol}/{symbol}_{timeframe}.json"
        try:
            path = hf_hub_download(
                repo_id=dataset_name,
                filename=filename,
                repo_type=repo_type,
                local_dir="./data"
            )
            downloaded_pairs.append((symbol, timeframe))
            print(f"  ✔ {symbol} {timeframe}")
        except Exception as e:
            pass

print(f"  ✔ 總共下載 {len(downloaded_pairs)} 個檔案")

# ======================== 第五步：模型訓練 ========================
print("\n[5/7] 模型訓練中...")

def create_model(lookback=60, future_steps=10):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(lookback, 9), return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(future_steps * 4)  # 10根K棒 * 4 (OHLC)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def prepare_sequences(df, lookback=60, future_steps=10):
    data = df[['open', 'high', 'low', 'close', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position']].values
    
    X, y = [], []
    
    for i in range(len(data) - lookback - future_steps + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+future_steps, :4].flatten())  # 只預測OHLC
    
    return np.array(X), np.array(y)

trained_count = 0

for symbol, timeframe in downloaded_pairs:
    print(f"\n  ▶ {symbol} {timeframe}")
    
    try:
        json_path = f"./data/klines_binance_us/{symbol}/{symbol}_{timeframe}.json"
        
        with open(json_path, 'r') as f:
            klines = json.load(f)
        
        df = pd.DataFrame(klines)
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) < 200:
            print(f"    ⚠ 資料不足")
            continue
        
        # 添加技術指標
        df = add_technical_indicators(df)
        df = df.iloc[30:].reset_index(drop=True)
        
        # 正規化
        scaler = MinMaxScaler()
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = scaler.fit_transform(df[price_cols])
        
        # 準備序列
        X, y = prepare_sequences(df, lookback=60, future_steps=10)
        
        # 分割訓練/驗證數據
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        # 建立並訓練模型
        model = create_model(lookback=60, future_steps=10)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=25,
            batch_size=16,
            verbose=0
        )
        
        # 儲存模型
        os.makedirs(f"./all_models/{symbol}", exist_ok=True)
        model_path = f"./all_models/{symbol}/{symbol}_{timeframe}_v8.keras"
        model.save(model_path)
        
        trained_count += 1
        print(f"    ✔ 訓練完成 - loss: {history.history['loss'][-1]:.6f}")
        
        # 釋放記憶體
        del model, X, y, X_train, y_train, X_val, y_val, df
        gc.collect()
        
    except Exception as e:
        print(f"    ⚠ {str(e)[:60]}")
        continue

print(f"\n  ✔ 成功訓練 {trained_count} 個檔案")

# ======================== 第六步：上傳到HF ========================
print("\n[6/7] 上傳模型到Hugging Face...")

try:
    from huggingface_hub import HfApi
    
    api = HfApi()
    upload_count = 0
    
    for symbol in os.listdir("./all_models"):
        symbol_path = f"./all_models/{symbol}"
        
        if not os.path.isdir(symbol_path):
            continue
        
        for model_file in os.listdir(symbol_path):
            if model_file.endswith('.keras'):
                local_path = os.path.join(symbol_path, model_file)
                repo_path = f"models_v8/{symbol}/{model_file}"
                
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=repo_path,
                        repo_id=dataset_name,
                        repo_type="dataset",
                        commit_message=f"Upload {symbol} {model_file}"
                    )
                    upload_count += 1
                    print(f"  ✔ {repo_path}")
                except Exception as e:
                    print(f"  ⚠ {symbol}/{model_file} - {str(e)[:40]}")
    
    print(f"  ✔ 成功上傳 {upload_count} 個檔案")
    
except Exception as e:
    print(f"  ⚠ 上傳失敗: {e}")
    print("  提示: 請確保已設定HF Token")

# ======================== 完成 ========================
print("\n[7/7] 完成")
print(f"\n" + "="*60)
print(f"訓練上傳完成！")
print(f"訓練模型: {trained_count}")
print(f"細誤時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
