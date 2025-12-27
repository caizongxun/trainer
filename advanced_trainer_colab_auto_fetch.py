#!/usr/bin/env python3
"""
Colab優化的進階虛擬貨幣價格預測模型訓練
自動從HuggingFace下載CSV數據並進行完整訓練
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

print("[1/8] Colab環境設定...")

try:
    from google.colab import drive
    IS_COLAB = True
    print("  ✔ Google Colab環境偵測成功")
except ImportError:
    IS_COLAB = False
    print("  ⚠ 本地環境模式")

print("  GPU優化配置...")
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

print("\n[2/8] 安裝依賴套件...")

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
print("\n[3/8] 技術指標計算...")

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

# ======================== 第四步：從HF自動下載數據 ========================
print("\n[4/8] 資料下載扣調...")

os.makedirs("./data/klines_binance_us", exist_ok=True)
os.makedirs("./all_models", exist_ok=True)

from huggingface_hub import list_repo_files, hf_hub_download

DATASET_ID = "zongowo111/cpb-models"
DATA_DIR = "./data/klines_binance_us"

print(f"  從HF下載: {DATASET_ID}")

try:
    # 列出HF上的所有檔案
    files = list_repo_files(repo_id=DATASET_ID, repo_type="dataset", revision="main")
    
    # 篩選出klines_binance_us資料夾中的CSV檔案
    csv_files = [f for f in files if f.startswith('klines_binance_us/') and f.endswith('.csv')]
    
    print(f"  ✔ 找到 {len(csv_files)} 個CSV檔案")
    
    # 建立符號資料夾
    symbols = {}
    for csv_file in csv_files:
        parts = csv_file.split('/')
        if len(parts) == 3:  # klines_binance_us/SYMBOL/file.csv
            symbol = parts[1]
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(csv_file)
    
    print(f"  ✔ 找到 {len(symbols)} 個幣種")
    
    # 下載檔案
    downloaded_count = 0
    pairs_to_train = []
    
    for symbol in sorted(symbols.keys()):
        symbol_path = f"{DATA_DIR}/{symbol}"
        os.makedirs(symbol_path, exist_ok=True)
        
        for csv_file in symbols[symbol]:
            filename = csv_file.split('/')[-1]
            local_path = f"{symbol_path}/{filename}"
            
            # 如果檔案不存在才下載
            if not os.path.exists(local_path):
                try:
                    print(f"    下載 {csv_file}...", end=' ')
                    hf_hub_download(
                        repo_id=DATASET_ID,
                        filename=csv_file,
                        repo_type="dataset",
                        local_dir="./data"
                    )
                    downloaded_count += 1
                    print("✔")
                except Exception as e:
                    print(f"✗ ({str(e)[:30]})")
            else:
                downloaded_count += 1
            
            # 記錄訓練對
            timeframe = filename.split('_')[1] if '_' in filename else 'unknown'
            pairs_to_train.append((symbol, timeframe, local_path))
    
    print(f"  ✔ 總共下載 {downloaded_count} 個檔案")
    print(f"  ✔ 準備訓練 {len(pairs_to_train)} 個模型")
    
except Exception as e:
    print(f"  ✗ 下載失敗: {e}")
    print("  嘗試使用測試數據...")
    pairs_to_train = [("BTC", "15m", None), ("ETH", "15m", None)]

# ======================== 第五步：模型訓練 ========================
print("\n[5/8] 模型訓練中...")

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

# 訓練前20個模型
for idx, (symbol, timeframe, csv_path) in enumerate(pairs_to_train[:20], 1):
    print(f"\n  [{idx}/{min(20, len(pairs_to_train))}] {symbol} {timeframe}")
    
    try:
        # 載入數據
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 標準化欄位名稱
            df.columns = [col.lower().strip() for col in df.columns]
            
            # 確保必要欄位存在
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                # 嘗試第一列作為時間索引
                df = df.iloc[:, 1:]
                df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        else:
            # 生成測試數據
            np.random.seed(42 + hash(symbol + timeframe) % 1000)
            df = pd.DataFrame({
                'open': np.random.uniform(0.9, 1.1, 10000).cumprod() * 100,
                'high': np.random.uniform(1.0, 1.15, 10000).cumprod() * 100,
                'low': np.random.uniform(0.85, 1.0, 10000).cumprod() * 100,
                'close': np.random.uniform(0.9, 1.1, 10000).cumprod() * 100,
                'volume': np.random.uniform(1000, 10000, 10000)
            })
        
        # 數據類型轉換
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) < 200:
            print(f"    ⚠ 數據不足 ({len(df)} < 200)")
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
        loss = history.history['loss'][-1]
        print(f"    ✔ 訓練完成 - loss: {loss:.6f}")
        
        # 釋放記憶體
        del model, X, y, X_train, y_train, X_val, y_val, df
        gc.collect()
        
    except Exception as e:
        print(f"    ✗ 訓練失敗: {str(e)[:60]}")
        continue

print(f"\n  ✔ 成功訓練 {trained_count} 個檔案")

# ======================== 第六步：上傳模型到Hugging Face ========================
print("\n[6/8] 上傳模型到Hugging Face...")

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
                    print(f"  ⚠ {repo_path} - {str(e)[:40]}")
    
    print(f"  ✔ 成功上傳 {upload_count} 個檔案")
    
except Exception as e:
    print(f"  ⚠ 上傳失敗: {e}")
    print("  提示: 請確保已設定HF Token")

# ======================== 第七步：生成摘要 ========================
print("\n[7/8] 生成訓練摘要...")

summary = {
    "timestamp": datetime.now().isoformat(),
    "trained_models": trained_count,
    "total_pairs": len(pairs_to_train),
    "dataset": DATASET_ID,
    "status": "completed"
}

with open("./training_summary.json", 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("  ✔ 摘要已保存到 training_summary.json")

# ======================== 完成 ========================
print("\n[8/8] 完成")
print(f"\n" + "="*60)
print("訓練上傳完成")
print(f"訓練模型: {trained_count}")
print(f"細誤時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
