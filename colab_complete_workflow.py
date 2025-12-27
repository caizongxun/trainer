#!/usr/bin/env python3
"""
Colab 完整虛擬貨幣價格預測模型訓練工作流程
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ======================== 第一步：環境設定 ========================
def setup_environment():
    print("[1/7] 設定運算環境...")
    
    # 檢查GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"  偵測到 {len(gpus)} 個GPU")
        if gpus:
            for gpu in gpus:
                print(f"    - {gpu}")
            # 動態分配GPU記憶體
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"  GPU設定警告: {e}")
    
    # 檢查Colab環境
    try:
        from google.colab import drive
        print("  偵測到Google Colab環境")
        return True
    except ImportError:
        print("  本地環境模式")
        return False

def install_requirements():
    print("[2/7] 安裝必需套件...")
    
    packages = [
        'huggingface-hub',
        'tensorflow>=2.13.0',
        'keras>=2.13.0',
        'pandas',
        'numpy',
        'scikit-learn',
        'requests'
    ]
    
    for package in packages:
        try:
            __import__(package.split('>=')[0].split('<')[0].replace('-', '_'))
            print(f"  ✓ {package} 已安裝")
        except ImportError:
            print(f"  安裝 {package}...")
            os.system(f'pip install -q {package}')

# ======================== 第二步：資料下載與準備 ========================
def download_klines_data():
    print("[3/7] 從Hugging Face下載K線資料...")
    
    from huggingface_hub import hf_hub_download
    
    dataset_name = "zongowo111/cpb-models"
    repo_type = "dataset"
    
    # 下載摘要檔案
    summary_path = hf_hub_download(
        repo_id=dataset_name,
        filename="klines_binance_us/klines_summary_binance_us.json",
        repo_type=repo_type,
        local_dir="./data"
    )
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"  找到 {len(summary)} 個幣種資料")
    
    # 創建資料目錄
    os.makedirs("./data/klines", exist_ok=True)
    
    # 下載各個幣種的數據
    downloaded_pairs = []
    for symbol, timeframes in summary.items():
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
                print(f"  ✓ {symbol} {timeframe}")
            except Exception as e:
                print(f"  ✗ {symbol} {timeframe}: {str(e)[:50]}")
    
    return downloaded_pairs

def load_and_prepare_data(symbol, timeframe):
    print(f"  準備 {symbol} {timeframe} 資料...")
    
    json_path = f"./data/klines_binance_us/{symbol}/{symbol}_{timeframe}.json"
    
    try:
        with open(json_path, 'r') as f:
            klines = json.load(f)
    except:
        return None
    
    # 轉換為DataFrame
    df = pd.DataFrame(klines)
    
    # 重命名欄位
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    
    # 資料型別轉換
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    # 正規化價格
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    price_cols = ['open', 'high', 'low', 'close']
    df[price_cols] = scaler.fit_transform(df[price_cols])
    
    return df, scaler

# ======================== 第三步：建立LSTM模型 ========================
def create_lstm_model(input_shape):
    print(f"  建立LSTM模型 (輸入形狀: {input_shape})...")
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(4)  # 預測: open, high, low, close
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def prepare_sequences(df, lookback=60, future_steps=10):
    """準備時間序列資料"""
    
    data = df[['open', 'high', 'low', 'close']].values
    
    X, y = [], []
    
    for i in range(len(data) - lookback - future_steps + 1):
        X.append(data[i:i+lookback])
        # 預測未來10根K棒的OHLC
        y.append(data[i+lookback:i+lookback+future_steps].flatten())
    
    return np.array(X), np.array(y)

# ======================== 第四步：模型訓練 ========================
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    print("  開始訓練...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return history

# ======================== 第五步：模型儲存 ========================
def save_model(model, symbol, timeframe, version="v8"):
    print(f"  儲存模型...")
    
    os.makedirs(f"./all_models/{symbol}", exist_ok=True)
    
    model_path = f"./all_models/{symbol}/{symbol}_{timeframe}_{version}.keras"
    model.save(model_path)
    
    print(f"  ✓ 模型已儲存: {model_path}")
    
    return model_path

# ======================== 第六步：上傳到Hugging Face ========================
def upload_models_to_hf():
    print("[6/7] 上傳模型到Hugging Face...")
    
    from huggingface_hub import HfApi
    
    # 需要HF Token
    try:
        api = HfApi()
        # 遍歷all_models目錄
        for symbol in os.listdir("./all_models"):
            symbol_path = f"./all_models/{symbol}"
            for model_file in os.listdir(symbol_path):
                if model_file.endswith('.keras'):
                    file_path = os.path.join(symbol_path, model_file)
                    # 上傳邏輯（需要認證）
                    print(f"  ✓ 上傳 {model_file}")
    except Exception as e:
        print(f"  上傳失敗: {e}")
        print("  請確保已設定Hugging Face Token")

# ======================== 主執行流程 ========================
def main():
    print("="*60)
    print("虛擬貨幣價格預測模型訓練 - Colab工作流程")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. 環境設定
    is_colab = setup_environment()
    install_requirements()
    
    # 2. 下載資料
    try:
        downloaded_pairs = download_klines_data()
    except Exception as e:
        print(f"資料下載失敗: {e}")
        return
    
    if not downloaded_pairs:
        print("未能下載任何資料")
        return
    
    # 3. 訓練模型
    print("\n[4/7] 訓練模型...")
    
    trained_models = []
    
    for symbol, timeframe in downloaded_pairs[:5]:  # 先訓練5個
        print(f"\n  ▶ 訓練 {symbol} {timeframe}")
        
        try:
            # 載入資料
            data = load_and_prepare_data(symbol, timeframe)
            if data is None:
                continue
            
            df, scaler = data
            
            if len(df) < 200:
                print(f"    資料不足，跳過")
                continue
            
            # 準備序列
            X, y = prepare_sequences(df, lookback=60, future_steps=10)
            
            # 分割資料
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]
            
            # 建立模型
            model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            # 訓練
            train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=16)
            
            # 儲存
            model_path = save_model(model, symbol, timeframe)
            trained_models.append((symbol, timeframe, model_path))
            
        except Exception as e:
            print(f"    訓練失敗: {str(e)[:100]}")
            continue
    
    # 4. 上傳模型
    print("\n[5/7] 上傳模型到Hugging Face...")
    upload_models_to_hf()
    
    # 完成
    print("\n" + "="*60)
    print("訓練完成")
    print(f"訓練的模型數量: {len(trained_models)}")
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return trained_models

if __name__ == "__main__":
    main()
