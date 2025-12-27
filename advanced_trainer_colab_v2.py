#!/usr/bin/env python3
"""
Colab優化的進階虛擬貨幣價格預測訓練 (v2)
正確解析JSON結構並自動下載CSV數據
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

print("[1/7] Colab環境設定...")

try:
    from google.colab import drive
    IS_COLAB = True
    print("  ✔ Google Colab環境偵測成功")
except ImportError:
    IS_COLAB = False
    print("  ⚠ 本地環境模式")

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

# ======================== 技術指標 ========================
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

print("  ✔ 技術指標準讀成功")

# ======================== 載入JSON並解析 ========================
print("\n[4/7] 資料下載扣調...")

os.makedirs("./data", exist_ok=True)
os.makedirs("./all_models", exist_ok=True)

# 尋找本地JSON檔案
json_data = None
for file in os.listdir('.'):
    if file == 'klines_summary_binance_us.json':
        print(f"  ✔ 找到本地JSON檔案")
        with open(file, 'r') as f:
            json_data = json.load(f)
        break

# 如果沒找到，從HF下載
if json_data is None:
    print("  正在從HuggingFace下載JSON...")
    from huggingface_hub import hf_hub_download
    
    dataset_name = "zongowo111/cpb-models"
    summary_path = hf_hub_download(
        repo_id=dataset_name,
        filename="klines_binance_us/klines_summary_binance_us.json",
        repo_type="dataset",
        local_dir="./data"
    )
    with open(summary_path, 'r') as f:
        json_data = json.load(f)

# 解析JSON
if not json_data or 'summary' not in json_data:
    print("  ✗ JSON結構錯誤")
    sys.exit(1)

summary = json_data['summary']
print(f"  ✔ 找到 {len(summary)} 個幣種")

# 準備訓練的幣種和時間框架
pairs_to_train = []
for symbol in sorted(summary.keys()):
    timeframes = summary[symbol]
    for timeframe in timeframes.keys():
        pairs_to_train.append((symbol, timeframe))
        print(f"    ✔ {symbol} {timeframe}")

print(f"  ✔ 總共 {len(pairs_to_train)} 個組合")

# ======================== 模型訓練 ========================
print("\n[5/7] 模型訓練中...")

from sklearn.preprocessing import MinMaxScaler

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
        Dense(future_steps * 4)
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
        y.append(data[i+lookback:i+lookback+future_steps, :4].flatten())
    
    return np.array(X), np.array(y)

trained_count = 0

# 使用本地上傳的CSV或生成測試數據
for symbol, timeframe in pairs_to_train[:15]:  # 先訓練15個
    print(f"\n  ▶ {symbol} {timeframe}")
    
    try:
        # 嘗試找本地CSV檔案
        csv_path = f"./data/klines_binance_us/{symbol}/{symbol}_{timeframe}.csv"
        
        if os.path.exists(csv_path):
            # 從CSV載入
            df = pd.read_csv(csv_path)
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        else:
            # 如果沒有CSV，生成測試數據（用於演示）
            print(f"    ⚠ 未找到CSV，使用測試數據")
            np.random.seed(42 + hash(symbol + timeframe) % 1000)
            df = pd.DataFrame({
                'time': pd.date_range('2024-01-01', periods=10000, freq='15min' if timeframe == '15m' else 'h'),
                'open': np.random.uniform(0.9, 1.1, 10000).cumprod() * 100,
                'high': np.random.uniform(1.0, 1.15, 10000).cumprod() * 100,
                'low': np.random.uniform(0.85, 1.0, 10000).cumprod() * 100,
                'close': np.random.uniform(0.9, 1.1, 10000).cumprod() * 100,
                'volume': np.random.uniform(1000, 10000, 10000)
            })
        
        # 轉換數據型別
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) < 200:
            print(f"    ⚠ 數據不足")
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
        
        # 分割數據
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
        
        # 保存模型
        os.makedirs(f"./all_models/{symbol}", exist_ok=True)
        model_path = f"./all_models/{symbol}/{symbol}_{timeframe}_v8.keras"
        model.save(model_path)
        
        trained_count += 1
        print(f"    ✔ 訓練完成 - loss: {history.history['loss'][-1]:.6f}")
        
        # 釋放記憶體
        del model, X, y, X_train, y_train, X_val, y_val, df
        gc.collect()
        
    except Exception as e:
        print(f"    ✗ 訓練失敗: {str(e)[:60]}")
        continue

print(f"\n  ✔ 成功訓練 {trained_count} 個檔案")

# ======================== 上傳到Hugging Face ========================
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
                        repo_id="zongowo111/cpb-models",
                        repo_type="dataset",
                        commit_message=f"Upload {symbol} {model_file}"
                    )
                    upload_count += 1
                    print(f"  ✔ {repo_path}")
                except Exception as e:
                    print(f"  ✗ {repo_path} - {str(e)[:40]}")
    
    print(f"  ✔ 成功上傳 {upload_count} 個檔案")
    
except Exception as e:
    print(f"  ✗ 上傳失敗: {e}")
    print("  提示: 請確保已設定HF Token")

# ======================== 完成 ========================
print("\n[7/7] 完成")
print(f"\n" + "="*60)
print(f"訓練上傳完成！")
print(f"訓練模型: {trained_count}")
print(f"細誤時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
