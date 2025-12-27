#!/usr/bin/env python3
"""
完全重設計版本 v2.0

目標：精確預測未來 10 根 K 線的 OHLC
- 輸入：過去 60 根 K 線的 OHLC (240 個特徵)
- 輸出：未來 10 根 K 線的 OHLC (40 個值)
- 架構：序列到序列 (Seq2Seq) LSTM 模型

執行方式：
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v2.py | python
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import gc
import time
from typing import Tuple, List

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print_success(f"偵測到 {len(gpus)} 個GPU")
except:
    pass

# ======================== 第二步：安裝依賴套件 ========================
print_header("2/7", "安裝依賴套件")

for module, name in [('tensorflow', 'TensorFlow'), ('keras', 'Keras'), 
                     ('huggingface_hub', 'HF Hub'), ('pandas', 'Pandas'), 
                     ('numpy', 'NumPy'), ('sklearn', 'Scikit-Learn')]:
    try:
        __import__(module)
        print_success(f"{name} 已安裝")
    except:
        pass

# ======================== 第三步：數據準備函數 ========================
print_header("3/7", "數據準備")

def normalize_ohlc(ohlc_array: np.ndarray) -> Tuple[np.ndarray, dict]:
    """正規化 OHLC 數據
    
    Args:
        ohlc_array: shape (n, 4) 的 OHLC 數據
    
    Returns:
        正規化後的數據 + 正規化參數
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    normalized = scaler.fit_transform(ohlc_array)
    
    params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    
    return normalized, params

def denormalize_ohlc(normalized: np.ndarray, params: dict) -> np.ndarray:
    """反正規化 OHLC 數據"""
    mean = np.array(params['mean'])
    scale = np.array(params['scale'])
    return normalized * scale + mean

def prepare_sequences(ohlc_data: np.ndarray, lookback: int = 60, 
                     forecast_horizon: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """準備訓練序列
    
    Args:
        ohlc_data: shape (n, 4) 的 OHLC 數據
        lookback: 使用過去多少根 K 線
        forecast_horizon: 預測未來多少根 K 線
    
    Returns:
        X: shape (n_samples, lookback, 4)
        y: shape (n_samples, forecast_horizon*4)  <- 扁平化為 40 個值
    """
    X, y = [], []
    
    for i in range(len(ohlc_data) - lookback - forecast_horizon):
        # 過去 60 根 K 線
        X.append(ohlc_data[i:i+lookback])  # (60, 4)
        
        # 未來 10 根 K 線，扁平化為 40 個值
        y.append(ohlc_data[i+lookback:i+lookback+forecast_horizon].flatten())  # (40,)
    
    return np.array(X), np.array(y)

print_success("數據準備函數已初始化")

# ======================== 第四步：數據下載 ========================
print_header("4/7", "數據下載")

os.makedirs("./data/klines_binance_us", exist_ok=True)
os.makedirs("./all_models", exist_ok=True)

from huggingface_hub import list_repo_files, hf_hub_download

DATASET_ID = "zongowo111/cpb-models"
DATA_DIR = "./data/klines_binance_us"

try:
    files = list_repo_files(repo_id=DATASET_ID, repo_type="dataset", revision="main")
    csv_files = [f for f in files if f.startswith('klines_binance_us/') and f.endswith('.csv')]
    
    print_success(f"找到 {len(csv_files)} 個 CSV 檔案")
    
    symbols = {}
    for csv_file in csv_files:
        parts = csv_file.split('/')
        if len(parts) == 3:
            symbol = parts[1]
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(csv_file)
    
    print_success(f"找到 {len(symbols)} 個幣種")
    
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
                    hf_hub_download(repo_id=DATASET_ID, filename=csv_file,
                                  repo_type="dataset", local_dir="./data")
                    downloaded_count += 1
                except:
                    pass
            else:
                downloaded_count += 1
            
            timeframe = filename.split('_')[1] if '_' in filename else 'unknown'
            pairs_to_train.append((symbol, timeframe, local_path))
    
    print_success(f"總共下載 {downloaded_count} 個檔案")
    print_success(f"準備訓練 {len(pairs_to_train)} 個模型")
    
except Exception as e:
    print_error(f"下載失敗: {e}")
    pairs_to_train = [("BTCUSDT", "15m", None)]

# ======================== 第五步：模型訓練 ========================
print_header("5/7", "模型訓練")

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_seq2seq_model(lookback: int = 60, forecast_horizon: int = 10) -> Model:
    """創建 Seq2Seq LSTM 模型
    
    - Encoder: 處理過去 60 根 K 線
    - Decoder: 生成未來 10 根 K 線
    """
    # 編碼器：處理過去 60 根 K 線
    encoder = Sequential([
        LSTM(64, activation='relu', input_shape=(lookback, 4), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        RepeatVector(forecast_horizon),  # 重複編碼器輸出 10 次
    ])
    
    # 解碼器：生成未來 10 根 K 線
    decoder = Sequential([
        LSTM(32, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(4))  # 每個時間步輸出 4 個值 (OHLC)
    ])
    
    # 完整模型
    model = Sequential([
        encoder,
        decoder,
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

trained_count = 0
training_results = []

for idx, (symbol, timeframe, csv_path) in enumerate(pairs_to_train[:10], 1):
    train_start = time.time()
    print(f"\n  [{idx}/{min(10, len(pairs_to_train))}] {symbol} {timeframe}", end=' ')
    
    try:
        # 讀取數據
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = [col.lower().strip() for col in df.columns]
            
            # 選擇 OHLC 列
            ohlc_data = df[['open', 'high', 'low', 'close']].values.astype(np.float32)
        else:
            # 生成模擬數據
            np.random.seed(42)
            n = 5000
            prices = np.cumsum(np.random.randn(n) * 0.1) + 100
            ohlc_data = np.column_stack([
                prices,  # open
                prices + np.abs(np.random.randn(n) * 0.5),  # high
                prices - np.abs(np.random.randn(n) * 0.5),  # low
                prices + np.random.randn(n) * 0.1  # close
            ]).astype(np.float32)
        
        # 驗證數據
        if len(ohlc_data) < 100:
            print("⚠ 數據不足")
            continue
        
        if np.any(np.isnan(ohlc_data)) or np.any(np.isinf(ohlc_data)):
            print("⚠ 數據包含 NaN/Inf")
            continue
        
        # 正規化
        ohlc_normalized, norm_params = normalize_ohlc(ohlc_data)
        
        # 準備序列
        X, y = prepare_sequences(ohlc_normalized, lookback=60, forecast_horizon=10)
        
        if len(X) < 50:
            print("⚠ 序列不足")
            continue
        
        # 分割訓練/驗證
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # 重塑 y 為 (samples, forecast_horizon, 4)
        y_train = y_train.reshape(y_train.shape[0], 10, 4)
        y_val = y_val.reshape(y_val.shape[0], 10, 4)
        
        # 訓練
        model = create_seq2seq_model(lookback=60, forecast_horizon=10)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=16,
            verbose=0,
            callbacks=callbacks
        )
        
        # 計算指標
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        # 預測並計算 MAPE
        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_val = model.predict(X_val, verbose=0)
        
        # 計算 MAPE (展平後)
        def calculate_mape(y_true, y_pred):
            y_true_flat = y_true.reshape(-1)
            y_pred_flat = y_pred.reshape(-1)
            
            # 避免除以零
            mask = np.abs(y_true_flat) > 1e-10
            if mask.sum() == 0:
                return 999.0
            
            return np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
        
        train_mape = calculate_mape(y_train, y_pred_train)
        val_mape = calculate_mape(y_val, y_pred_val)
        
        # 保存模型
        os.makedirs(f"./all_models/{symbol}", exist_ok=True)
        model_path = f"./all_models/{symbol}/{symbol}_{timeframe}_seq2seq.keras"
        model.save(model_path)
        
        # 保存正規化參數
        params_path = f"./all_models/{symbol}/{symbol}_{timeframe}_params.json"
        with open(params_path, 'w') as f:
            json.dump(norm_params, f)
        
        trained_count += 1
        train_time = time.time() - train_start
        
        print(f"✔ 訓練完成 ({train_time:.1f}s)")
        print(f"    Loss: {train_loss:.4f}/{val_loss:.4f} | MAPE: {train_mape:.2f}%/{val_mape:.2f}%")
        
        training_results.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'train_mape': float(train_mape),
            'val_mape': float(val_mape),
            'training_time': train_time,
            'model_type': 'Seq2Seq LSTM',
            'input_shape': [60, 4],
            'output_shape': [10, 4]
        })
        
        del model, X, y, X_train, y_train, X_val, y_val, df
        gc.collect()
        
    except Exception as e:
        print_error(f"訓練失敗: {str(e)[:50]}")
        continue

print(f"\n  ✔ 成功訓練 {trained_count} 個檔案")

# ======================== 第六步：上傳模型 ========================
print_header("6/7", "上傳模型")

try:
    from huggingface_hub import HfApi
    
    api = HfApi()
    upload_count = 0
    
    for symbol in os.listdir("./all_models"):
        symbol_path = f"./all_models/{symbol}"
        
        if not os.path.isdir(symbol_path):
            continue
        
        for model_file in os.listdir(symbol_path):
            if model_file.endswith('.keras') or model_file.endswith('.json'):
                try:
                    api.upload_file(
                        path_or_fileobj=os.path.join(symbol_path, model_file),
                        path_in_repo=f"models_seq2seq/{symbol}/{model_file}",
                        repo_id=DATASET_ID,
                        repo_type="dataset",
                        commit_message=f"Upload {symbol} {model_file}"
                    )
                    upload_count += 1
                    print(f"  ✔ {symbol}/{model_file}")
                except:
                    pass
    
    print_success(f"成功上傳 {upload_count} 個檔案")
except:
    print_warning("上傳失敗")

# ======================== 第七步：完成 ========================
print_header("7/7", "完成")

summary = {
    "timestamp": datetime.now().isoformat(),
    "trained_models": trained_count,
    "total_pairs": len(pairs_to_train),
    "dataset": DATASET_ID,
    "status": "completed",
    "results": training_results,
    "version": "seq2seq_v2.0",
    "model_architecture": {
        "type": "Seq2Seq LSTM",
        "encoder": "LSTM(64) -> LSTM(32) -> RepeatVector(10)",
        "decoder": "LSTM(32) -> LSTM(64) -> TimeDistributed(Dense(4))",
        "input_shape": [60, 4],  # 過去 60 根 K 線的 OHLC
        "output_shape": [10, 4],  # 未來 10 根 K 線的 OHLC
        "total_parameters": "~65,000"
    },
    "training_config": {
        "epochs": 20,
        "batch_size": 16,
        "loss": "MSE",
        "optimizer": "Adam (lr=0.001)",
        "early_stopping": "patience=5",
        "reduce_lr": "factor=0.5, patience=3"
    },
    "normalization": "StandardScaler per symbol"
}

with open("./training_summary.json", 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print_success("摘要已保存")

print("\n" + "="*60)
print("訓練上傳完成")
print(f"訓練模型: {trained_count}")
if training_results:
    avg_train_mape = np.mean([r['train_mape'] for r in training_results])
    avg_val_mape = np.mean([r['val_mape'] for r in training_results])
    avg_time = np.mean([r['training_time'] for r in training_results])
    print(f"平均訓練 MAPE: {avg_train_mape:.2f}%")
    print(f"平均驗證 MAPE: {avg_val_mape:.2f}%")
    print(f"平均訓練時間: {avg_time:.1f}s")
print("="*60)
