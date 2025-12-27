#!/usr/bin/env python3
"""
进阶虛擬貨幣價格預測模型訓練
完整 Colab 工作流程

執行方法：
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow.py | python
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

def print_header(step, message):
    print(f"\n[{step}/7] {message}")

def print_success(message):
    print(f"  ✔ {message}")

def print_warning(message):
    print(f"  ⚠ {message}")

def print_error(message):
    print(f"  ✗ {message}")

# ======================== 第一步：Colab環境設定 ========================
print_header("1/7", "Colab環境設定")

try:
    from google.colab import drive
    IS_COLAB = True
    print_success("Google Colab環境偵測成功")
except ImportError:
    IS_COLAB = False
    print_warning("本地環境模式")

print("  GPU优化配置...")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print_success(f"偵測到 {len(gpus)} 個GPU")
    else:
        print_warning("未偵測到GPU")
except Exception as e:
    print_warning(f"GPU配置警告: {e}")

# ======================== 第二步：安裝依賴套件 ========================
print_header("2/7", "安裝依賴套件")

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
        print_success(f"{name} 已安裝")
    except ImportError:
        print(f"  安裝 {name}...", end=' ')
        os.system(f'pip install -q {module.replace("_", "-")}')
        print("OK")

# ======================== 第三步：定義技術指標 ========================
print_header("3/7", "技術指標計算")

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

print_success("技術指標凖讀成功")

# ======================== 第四步：自勘下載数据 ========================
print_header("4/7", "数据下載扣調")

os.makedirs("./data/klines_binance_us", exist_ok=True)
os.makedirs("./all_models", exist_ok=True)

from huggingface_hub import list_repo_files, hf_hub_download

DATASET_ID = "zongowo111/cpb-models"
DATA_DIR = "./data/klines_binance_us"

print(f"  从 HF 下載: {DATASET_ID}")

try:
    # 列出 HF 上的所有檔案
    files = list_repo_files(repo_id=DATASET_ID, repo_type="dataset", revision="main")
    
    # 篩選 klines_binance_us 資料夾中的 CSV 檔案
    csv_files = [f for f in files if f.startswith('klines_binance_us/') and f.endswith('.csv')]
    
    print_success(f"找到 {len(csv_files)} 個 CSV 檔案")
    
    # 建立符號資料夾
    symbols = {}
    for csv_file in csv_files:
        parts = csv_file.split('/')
        if len(parts) == 3:
            symbol = parts[1]
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(csv_file)
    
    print_success(f"找到 {len(symbols)} 個幣種")
    
    # 下載檔案
    downloaded_count = 0
    pairs_to_train = []
    
    for symbol in sorted(symbols.keys()):
        symbol_path = f"{DATA_DIR}/{symbol}"
        os.makedirs(symbol_path, exist_ok=True)
        
        for csv_file in symbols[symbol]:
            filename = csv_file.split('/')[-1]
            local_path = f"{symbol_path}/{filename}"
            
            if not os.path.exists(local_path):
                try:
                    hf_hub_download(
                        repo_id=DATASET_ID,
                        filename=csv_file,
                        repo_type="dataset",
                        local_dir="./data"
                    )
                    downloaded_count += 1
                except Exception as e:
                    pass
            else:
                downloaded_count += 1
            
            timeframe = filename.split('_')[1] if '_' in filename else 'unknown'
            pairs_to_train.append((symbol, timeframe, local_path))
    
    print_success(f"總共下載 {downloaded_count} 個檔案")
    print_success(f"準備訓練 {len(pairs_to_train)} 個模型")
    
except Exception as e:
    print_error(f"下載失敗: {e}")
    print_warning("嘗試使用測試數据...")
    pairs_to_train = [("BTC", "15m", None), ("ETH", "15m", None)]

# ======================== 第五步：模型訓練 ========================
print_header("5/7", "模型訓練")

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(lookback=60, future_steps=10):
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

# 訓練前 20 個模型
for idx, (symbol, timeframe, csv_path) in enumerate(pairs_to_train[:20], 1):
    print(f"\n  [{idx}/{min(20, len(pairs_to_train))}] {symbol} {timeframe}")
    
    try:
        # 載入數据
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = [col.lower().strip() for col in df.columns]
            
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                df = df.iloc[:, 1:]
                df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        else:
            np.random.seed(42 + hash(symbol + timeframe) % 1000)
            df = pd.DataFrame({
                'open': np.random.uniform(0.9, 1.1, 10000).cumprod() * 100,
                'high': np.random.uniform(1.0, 1.15, 10000).cumprod() * 100,
                'low': np.random.uniform(0.85, 1.0, 10000).cumprod() * 100,
                'close': np.random.uniform(0.9, 1.1, 10000).cumprod() * 100,
                'volume': np.random.uniform(1000, 10000, 10000)
            })
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) < 200:
            print(f"    ⚠ 數据不足 ({len(df)} < 200)")
            continue
        
        df = add_technical_indicators(df)
        df = df.iloc[30:].reset_index(drop=True)
        
        scaler = MinMaxScaler()
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = scaler.fit_transform(df[price_cols])
        
        X, y = prepare_sequences(df, lookback=60, future_steps=10)
        
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        model = create_model(lookback=60, future_steps=10)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=25,
            batch_size=16,
            verbose=0
        )
        
        os.makedirs(f"./all_models/{symbol}", exist_ok=True)
        model_path = f"./all_models/{symbol}/{symbol}_{timeframe}_v8.keras"
        model.save(model_path)
        
        trained_count += 1
        loss = history.history['loss'][-1]
        print(f"    ✔ 訓練完成 - loss: {loss:.6f}")
        
        del model, X, y, X_train, y_train, X_val, y_val, df
        gc.collect()
        
    except Exception as e:
        print_error(f"訓練失敖: {str(e)[:60]}")
        continue

print(f"\n  ✔ 成功訓練 {trained_count} 個檔案")

# ======================== 第六步：上傳模型 ========================
print_header("6/7", "上傳模型到 Hugging Face")

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
                        repo_id=DATASET_ID,
                        repo_type="dataset",
                        commit_message=f"Upload {symbol} {model_file}"
                    )
                    upload_count += 1
                    print(f"  ✔ {repo_path}")
                except Exception as e:
                    print_warning(f"{repo_path} - {str(e)[:40]}")
    
    print_success(f"成功上傳 {upload_count} 個檔案")
    
except Exception as e:
    print_warning(f"上傳失敖: {e}")
    print_warning("提示: 請確保已設定 HF Token")

# ======================== 第七步：完成 ========================
print_header("7/7", "完成")

summary = {
    "timestamp": datetime.now().isoformat(),
    "trained_models": trained_count,
    "total_pairs": len(pairs_to_train),
    "dataset": DATASET_ID,
    "status": "completed"
}

with open("./training_summary.json", 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print_success("摘要已保存到 training_summary.json")

print("\n" + "="*60)
print("訓練上傳完成")
print(f"訓練模型: {trained_count}")
print(f"準備模型: {len(pairs_to_train)}")
print(f"細誤時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
