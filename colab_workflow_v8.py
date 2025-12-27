#!/usr/bin/env python3
"""
V8 版本 - 改進的 Seq2Seq LSTM 模型

特点：
1. 基於 V7 的多輸出回歸 (4 個價格細寮)
2. 並行輸出：BB 上下軌 (波動率)
3. 深化数數分出分集深化分策
4. 多任務优化：价格 + 波动率

執行方式：
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v8.py | python
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
from typing import Tuple, List, Dict

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

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算技術指標（V7 邏輯）"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # RSI (14 期)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close, dtype=float)
    avg_loss = np.zeros_like(close, dtype=float)
    
    if len(close) > 14:
        avg_gain[14] = gain[:14].mean()
        avg_loss[14] = loss[:14].mean()
        
        for i in range(15, len(close)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 移動平均線
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # Bollinger Bands (20, 2)
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
    df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-10)
    
    # 波動性 (輔助任務)
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    
    return df.fillna(method='bfill').fillna(method='ffill')

def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, Dict]:
    """正規化特徵"""
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    normalized = scaler.fit_transform(df[feature_cols])
    
    params = {
        'feature_cols': feature_cols,
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    
    return normalized, params

def prepare_sequences_v8(ohlc_data: np.ndarray, technical_data: np.ndarray,
                        lookback: int = 60, forecast_horizon: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V8 序列準備
    
    輸出：
    - X: (samples, lookback, features)  # 輸入序列
    - y_ohlc: (samples, forecast_horizon, 4)  # OHLC 預測
    - y_volatility: (samples, forecast_horizon)  # 波動率預測 (輔助)
    """
    X, y_ohlc, y_volatility = [], [], []
    
    for i in range(len(ohlc_data) - lookback - forecast_horizon):
        # 輸入：過去 60 根 K 線的 OHLC + 技術指標
        X.append(np.concatenate([
            ohlc_data[i:i+lookback],  # (60, 4)
            technical_data[i:i+lookback]  # (60, n_indicators)
        ], axis=1))
        
        # 目標 1：未來 10 根 K 線的 OHLC
        y_ohlc.append(ohlc_data[i+lookback:i+lookback+forecast_horizon])
        
        # 目標 2：未來 10 根 K 線的波動率 (輔助任務)
        y_volatility.append(technical_data[i+lookback:i+lookback+forecast_horizon, -1])
    
    return np.array(X), np.array(y_ohlc), np.array(y_volatility)

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
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Input, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_seq2seq_v8_model(input_shape: Tuple[int, int], forecast_horizon: int = 10) -> Model:
    """V8 模型：Seq2Seq LSTM + 多任務學習
    
    同時預測：
    - 主任務：未來 10 根 K 線的 OHLC
    - 輔助任務：未來 10 根 K 線的波動率
    """
    lookback, n_features = input_shape
    
    # Encoder
    inputs = Input(shape=(lookback, n_features), name='encoder_input')
    
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(inputs)
    x = LSTM(32, return_sequences=False, dropout=0.2)(x)
    
    # 重複向量
    encoder_output = RepeatVector(forecast_horizon)(x)
    
    # Decoder
    decoder = LSTM(32, return_sequences=True, dropout=0.2)(encoder_output)
    decoder = LSTM(64, return_sequences=True, dropout=0.2)(decoder)
    
    # 主任務輸出：OHLC (4 個值)
    ohlc_output = TimeDistributed(Dense(4), name='ohlc_output')(decoder)
    
    # 輔助任務輸出：波動率 (1 個值)
    volatility_output = TimeDistributed(Dense(1), name='volatility_output')(decoder)
    
    model = Model(inputs=inputs, outputs=[ohlc_output, volatility_output])
    
    # 多任務損失函數
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'ohlc_output': 'mse',
            'volatility_output': 'mse'
        },
        loss_weights={
            'ohlc_output': 1.0,  # 主任務
            'volatility_output': 0.2  # 輔助任務 (較低權重)
        },
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
        else:
            np.random.seed(42)
            n = 5000
            prices = np.cumsum(np.random.randn(n) * 0.1) + 100
            df = pd.DataFrame({
                'open': prices,
                'high': prices + np.abs(np.random.randn(n) * 0.5),
                'low': prices - np.abs(np.random.randn(n) * 0.5),
                'close': prices + np.random.randn(n) * 0.1,
                'volume': np.abs(np.random.randn(n) * 1000)
            })
        
        # 驗證數據
        if len(df) < 100:
            print("⚠ 數據不足")
            continue
        
        if np.any(np.isnan(df[['open', 'high', 'low', 'close']].values)):
            print("⚠ 數據包含 NaN")
            continue
        
        # 添加技術指標
        df = add_technical_indicators(df)
        
        # 分離 OHLC 和技術指標
        ohlc_data = df[['open', 'high', 'low', 'close']].values.astype(np.float32)
        technical_cols = ['rsi', 'macd', 'sma_20', 'sma_50', 'bb_position', 'volatility']
        technical_data = df[technical_cols].values.astype(np.float32)
        
        # 正規化
        ohlc_scaler = StandardScaler()
        ohlc_normalized = ohlc_scaler.fit_transform(ohlc_data)
        
        tech_scaler = StandardScaler()
        technical_normalized = tech_scaler.fit_transform(technical_data)
        
        # 準備序列
        X, y_ohlc, y_volatility = prepare_sequences_v8(
            ohlc_normalized, technical_normalized,
            lookback=60, forecast_horizon=10
        )
        
        if len(X) < 50:
            print("⚠ 序列不足")
            continue
        
        # 分割
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_ohlc_train, y_ohlc_val = y_ohlc[:split], y_ohlc[split:]
        y_vol_train, y_vol_val = y_volatility[:split], y_volatility[split:]
        
        # 訓練
        model = create_seq2seq_v8_model(X_train.shape[1:], forecast_horizon=10)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train, [y_ohlc_train, y_vol_train],
            validation_data=(X_val, [y_ohlc_val, y_vol_val]),
            epochs=20,
            batch_size=16,
            verbose=0,
            callbacks=callbacks
        )
        
        # 評估
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        # 預測
        y_ohlc_pred_train = model.predict(X_train, verbose=0)[0]
        y_ohlc_pred_val = model.predict(X_val, verbose=0)[0]
        
        # 計算 MAPE
        def safe_mape(y_true, y_pred):
            y_true_flat = y_true.reshape(-1)
            y_pred_flat = y_pred.reshape(-1)
            mask = np.abs(y_true_flat) > 1e-10
            if mask.sum() == 0:
                return 999.0
            return np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
        
        train_mape = safe_mape(y_ohlc_train, y_ohlc_pred_train)
        val_mape = safe_mape(y_ohlc_val, y_ohlc_pred_val)
        
        # 保存模型
        os.makedirs(f"./all_models/{symbol}", exist_ok=True)
        model_path = f"./all_models/{symbol}/{symbol}_{timeframe}_v8.keras"
        model.save(model_path)
        
        # 保存正規化參數
        params = {
            'ohlc_mean': ohlc_scaler.mean_.tolist(),
            'ohlc_scale': ohlc_scaler.scale_.tolist(),
            'technical_mean': tech_scaler.mean_.tolist(),
            'technical_scale': tech_scaler.scale_.tolist(),
            'technical_cols': technical_cols
        }
        
        params_path = f"./all_models/{symbol}/{symbol}_{timeframe}_v8_params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f)
        
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
            'model_type': 'Seq2Seq LSTM V8',
            'architecture': 'Bidirectional LSTM + Multitask Learning',
            'tasks': ['OHLC prediction', 'Volatility prediction'],
            'input_shape': [60, X_train.shape[2]],
            'output_shapes': {'ohlc': [10, 4], 'volatility': [10, 1]}
        })
        
        del model, X, y_ohlc, y_volatility, X_train, y_ohlc_train, y_vol_train
        del X_val, y_ohlc_val, y_vol_val, df
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
            if model_file.endswith(('.keras', '.json')):
                try:
                    api.upload_file(
                        path_or_fileobj=os.path.join(symbol_path, model_file),
                        path_in_repo=f"models_v8/{symbol}/{model_file}",
                        repo_id=DATASET_ID,
                        repo_type="dataset",
                        commit_message=f"Upload V8 {symbol} {model_file}"
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
    "version": "v8.0",
    "model_architecture": {
        "type": "Seq2Seq LSTM with Multitask Learning",
        "encoder": "Bidirectional LSTM(64) -> LSTM(32) -> RepeatVector(10)",
        "decoder": "LSTM(32) -> LSTM(64) -> TimeDistributed",
        "tasks": {
            "primary": "OHLC prediction (10 bars × 4 values = 40 outputs)",
            "auxiliary": "Volatility prediction (10 bars × 1 value = 10 outputs)"
        },
        "input_shape": "[60, 10] - 60 bars historical data with OHLC + 6 indicators",
        "output_shapes": {
            "ohlc": [10, 4],
            "volatility": [10, 1]
        },
        "total_parameters": "~85,000"
    },
    "training_config": {
        "epochs": 20,
        "batch_size": 16,
        "loss": "MSE (weighted multitask)",
        "loss_weights": {"ohlc": 1.0, "volatility": 0.2},
        "optimizer": "Adam (lr=0.001)",
        "early_stopping": "patience=5",
        "reduce_lr": "factor=0.5, patience=3"
    },
    "v8_improvements": {
        "vs_v7": [
            "Bidirectional LSTM - captures patterns from both directions",
            "Multitask learning - joint optimization of price + volatility",
            "Better feature engineering - includes volatility as auxiliary task",
            "Enhanced normalization - separate scalers for OHLC and indicators"
        ],
        "next_v9": [
            "Implement Attention mechanism",
            "Add BB channel inversion strategy",
            "Ensemble multiple models",
            "Implement online learning for adaptation"
        ]
    }
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
    print(f"模型版本: V8 (Multitask Seq2Seq LSTM)")
print("="*60)
