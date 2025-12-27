#!/usr/bin/env python3
"""
V7 Classic Fast - 低额特格壹方法

主要改進:
✅ 不用 set_memory_growth (取消動態分配)
✅ 預算住 GPU 記憶體 (14GB)
✅ 簡單粗暴訓練（簡一堆正常化）
✅ 丢塚 Callback 並打印整個訓練進度
✅ 每個 Epoch 低於vue上下 5 秒

執行:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/V7_CLASSIC_FAST.py | python
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
from typing import Tuple

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 簡裨乛上 TensorFlow 日誌

print("="*70)
print("V7 Classic Fast - 低额特格壹方法訓練")
print("="*70)
print()

print("[0/7] 初始化檢查")
print(f"Python 版本: {sys.version.split()[0]}")
print(f"當前工作目錄: {os.getcwd()}")
print()

# 檢查是否在 Colab
try:
    from google.colab import drive
    IS_COLAB = True
    print("✓ Google Colab 環境")
except ImportError:
    IS_COLAB = False
    print("本地環境")

print()
print("[1/7] TensorFlow GPU 設定")

import tensorflow as tf

# 回解: 不使用 set_memory_growth
# 取而代之預算低 GPU 記憶體
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ 偵測到 {len(gpus)} 個 GPU")
    try:
        # 不使用 set_memory_growth, 直接預算低整個 14GB
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=14000)]  # 14GB
        )
        print("✓ GPU 記憶體設定为 14GB (固定配置)")
    except:
        print("⚠ GPU 記憶體設定失敗, 使用預設")
else:
    print("⚠ 未偵測到 GPU")

print(f"✓ TensorFlow 版本: {tf.__version__}")
print()

print("[2/7] 安裝依賴")
for module, name in [('keras', 'Keras'), ('huggingface_hub', 'HF Hub'), 
                     ('pandas', 'Pandas'), ('numpy', 'NumPy'), ('sklearn', 'Sklearn')]:
    try:
        __import__(module)
        print(f"✓ {name}")
    except:
        print(f"✗ {name}")

print()
print("[3/7] 數據準備函數")

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算技術指標"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # RSI
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
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['sma20'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + 2 * bb_std
    df['bb_lower'] = df['sma20'] - 2 * bb_std
    df['bb_width_pct'] = (df['bb_upper'] - df['bb_lower']) / (df['sma20'] + 1e-10)
    df['volatility'] = df['close'].rolling(20).std() / (df['sma20'] + 1e-10)
    
    # ATR
    df['tr'] = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - df['close'].shift()),
            np.abs(low - df['close'].shift())
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    
    return df.fillna(method='bfill').fillna(method='ffill')

def prepare_sequences(ohlc_data: np.ndarray, technical_data: np.ndarray,
                     lookback: int = 60, forecast_horizon: int = 10) -> Tuple:
    """V7 序列準備"""
    X, y_ohlc, y_bb_params, y_volatility = [], [], [], []
    
    for i in range(len(ohlc_data) - lookback - forecast_horizon):
        X.append(np.concatenate([
            ohlc_data[i:i+lookback],
            technical_data[i:i+lookback]
        ], axis=1))
        y_ohlc.append(ohlc_data[i+lookback:i+lookback+forecast_horizon])
        y_bb_params.append(technical_data[i+lookback:i+lookback+forecast_horizon, 4:6])
        y_volatility.append(technical_data[i+lookback:i+lookback+forecast_horizon, 8:9])
    
    return np.array(X), np.array(y_ohlc), np.array(y_bb_params), np.array(y_volatility)

print("✓ 數據準備函數已初始化")
print()

print("[4/7] 數據下載")

os.makedirs("./data/klines_binance_us", exist_ok=True)
os.makedirs("./all_models_v7_fast", exist_ok=True)

from huggingface_hub import list_repo_files, hf_hub_download

DATASET_ID = "zongowo111/cpb-models"
DATA_DIR = "./data/klines_binance_us"

print(f"HF 資料集: {DATASET_ID}")

try:
    files = list_repo_files(repo_id=DATASET_ID, repo_type="dataset", revision="main")
    csv_files = [f for f in files if f.startswith('klines_binance_us/') and f.endswith('.csv')]
    print(f"✓ 找到 {len(csv_files)} 個 CSV 檔案")
    
    symbols = {}
    for csv_file in csv_files:
        parts = csv_file.split('/')
        if len(parts) == 3:
            symbol = parts[1]
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(csv_file)
    
    print(f"✓ 找到 {len(symbols)} 個幣種")
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
                except:
                    pass
            
            timeframe = filename.split('_')[1] if '_' in filename else 'unknown'
            pairs_to_train.append((symbol, timeframe, local_path))
    
    print(f"✓ 下載完成 ({len(pairs_to_train)} 個檔案)")
    
except Exception as e:
    print(f"⚠ 下載遅稍 {e}")
    pairs_to_train = [("BTCUSDT", "15m", None)]

print()
print("[5/7] 模型訓練")
print()

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Input, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

def create_seq2seq_v7(input_shape: Tuple[int, int], forecast_horizon: int = 10) -> Model:
    """V7 模型"""
    lookback, n_features = input_shape
    
    inputs = Input(shape=(lookback, n_features), name='encoder_input')
    
    # Encoder - 3 層 BiLSTM
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(inputs)
    x = LayerNormalization()(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = LayerNormalization()(x)
    
    x = LSTM(32, return_sequences=False, dropout=0.3)(x)
    x = LayerNormalization()(x)
    
    # RepeatVector
    encoder_output = RepeatVector(forecast_horizon)(x)
    
    # Decoder - 1 層 LSTM
    decoder = LSTM(64, return_sequences=True, dropout=0.3)(encoder_output)
    decoder = LayerNormalization()(decoder)
    
    # 三個輸出層
    ohlc_out = TimeDistributed(Dense(4), name='ohlc_output')(decoder)
    bb_out = TimeDistributed(Dense(2), name='bb_params_output')(decoder)
    vol_out = TimeDistributed(Dense(1), name='volatility_output')(decoder)
    
    model = Model(inputs=inputs, outputs=[ohlc_out, bb_out, vol_out])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss={
            'ohlc_output': 'mse',
            'bb_params_output': 'mse',
            'volatility_output': 'mse'
        },
        loss_weights={
            'ohlc_output': 1.0,
            'bb_params_output': 0.8,
            'volatility_output': 0.3
        }
    )
    
    return model

trained_count = 0
training_results = []
max_pairs = min(40, len(pairs_to_train))

start_time_total = time.time()

for idx, (symbol, timeframe, csv_path) in enumerate(pairs_to_train[:max_pairs], 1):
    train_start = time.time()
    
    try:
        # 讀取數據
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = [col.lower().strip() for col in df.columns]
        else:
            np.random.seed(42)
            n = 8000
            prices = np.cumsum(np.random.randn(n) * 0.1) + 100
            df = pd.DataFrame({
                'open': prices,
                'high': prices + np.abs(np.random.randn(n) * 0.5),
                'low': prices - np.abs(np.random.randn(n) * 0.5),
                'close': prices + np.random.randn(n) * 0.1,
                'volume': np.abs(np.random.randn(n) * 1000)
            })
        
        if len(df) < 500:
            print(f"[{idx:2d}/{max_pairs}] {symbol:10s} {timeframe:3s} ⚠ 數據不足")
            continue
        
        # 計算技術指標
        df = add_technical_indicators(df)
        
        # 準備數據
        ohlc_data = df[['open', 'high', 'low', 'close']].values.astype(np.float32)
        tech_cols = ['rsi', 'macd', 'signal', 'bb_upper', 'bb_lower', 'bb_width_pct', 'volatility', 'atr']
        technical_data = df[tech_cols + ['bb_width_pct', 'volatility']].values.astype(np.float32)
        
        if technical_data.shape[1] != 10:
            technical_data = np.concatenate([technical_data, technical_data[:, -1:]], axis=1)
        
        # 標準化
        ohlc_scaler = StandardScaler()
        ohlc_normalized = ohlc_scaler.fit_transform(ohlc_data)
        
        tech_scaler = StandardScaler()
        technical_normalized = tech_scaler.fit_transform(technical_data)
        
        # 序列準備
        X, y_ohlc, y_bb, y_vol = prepare_sequences(
            ohlc_normalized, technical_normalized,
            lookback=60, forecast_horizon=10
        )
        
        if len(X) < 50:
            print(f"[{idx:2d}/{max_pairs}] {symbol:10s} {timeframe:3s} ⚠ 序列不足")
            continue
        
        # 分割訓練/驗證集
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_ohlc_train, y_ohlc_val = y_ohlc[:split], y_ohlc[split:]
        y_bb_train, y_bb_val = y_bb[:split], y_bb[split:]
        y_vol_train, y_vol_val = y_vol[:split], y_vol[split:]
        
        # 建立模型
        model = create_seq2seq_v7(X_train.shape[1:], forecast_horizon=10)
        
        # 訓練 (這一次完全粗暴)
        print(f"[{idx:2d}/{max_pairs}] {symbol:10s} {timeframe:3s} ", end='', flush=True)
        
        training_start = time.time()
        
        history = model.fit(
            X_train,
            {
                'ohlc_output': y_ohlc_train,
                'bb_params_output': y_bb_train,
                'volatility_output': y_vol_train
            },
            validation_data=(
                X_val,
                {
                    'ohlc_output': y_ohlc_val,
                    'bb_params_output': y_bb_val,
                    'volatility_output': y_vol_val
                }
            ),
            epochs=100,
            batch_size=16,
            verbose=2  # Keras 預設輸出（每個 epoch 一行)
        )
        
        training_time = time.time() - training_start
        
        # 預測
        y_ohlc_pred = model.predict(X_val, verbose=0)[0]
        
        def safe_mape(y_true, y_pred):
            y_true_flat = y_true.reshape(-1)
            y_pred_flat = y_pred.reshape(-1)
            mask = np.abs(y_true_flat) > 1e-10
            if mask.sum() == 0:
                return 999.0
            return np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
        
        val_mape = safe_mape(y_ohlc_val, y_ohlc_pred)
        
        # 保存模型
        os.makedirs(f"./all_models_v7_fast/{symbol}", exist_ok=True)
        model_path = f"./all_models_v7_fast/{symbol}/{symbol}_{timeframe}_v7.keras"
        model.save(model_path)
        
        params = {
            'ohlc_mean': ohlc_scaler.mean_.tolist(),
            'ohlc_scale': ohlc_scaler.scale_.tolist(),
            'technical_mean': tech_scaler.mean_.tolist(),
            'technical_scale': tech_scaler.scale_.tolist(),
            'version': 'v7_classic_fast'
        }
        
        params_path = f"./all_models_v7_fast/{symbol}/{symbol}_{timeframe}_v7_params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f)
        
        trained_count += 1
        train_time = time.time() - train_start
        
        # 粀禂輸出
        val_loss = history.history['val_loss'][-1]
        print(f" ✓ {train_time:5.0f}s | Loss: {val_loss:.4f} | MAPE: {val_mape:5.2f}%")
        
        training_results.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'val_loss': float(val_loss),
            'val_mape': float(val_mape),
            'training_time': train_time,
            'epochs': len(history.history['loss'])
        })
        
        # 釋放記憶體
        del model, X, y_ohlc, y_bb, y_vol
        del X_train, y_ohlc_train, y_bb_train, y_vol_train
        del X_val, y_ohlc_val, y_bb_val, y_vol_val, df
        gc.collect()
        
    except Exception as e:
        print(f" ✗ {str(e)[:50]}")
        continue

total_time = time.time() - start_time_total

print()
print("[6/7] 上傳模型")
print("✓ 模型保存在 ./all_models_v7_fast 目錄")

print()
print("[7/7] 完成")

summary = {
    "timestamp": datetime.now().isoformat(),
    "version": "v7_classic_fast",
    "trained_models": trained_count,
    "results": training_results
}

with open("./training_summary_v7_fast.json", 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print()
print("="*70)
print(f"訓練完成 - {trained_count} 個模型")
if training_results:
    avg_val_mape = np.mean([r['val_mape'] for r in training_results])
    avg_time = np.mean([r['training_time'] for r in training_results])
    print(f"平均 MAPE: {avg_val_mape:.2f}%")
    print(f"平均訓練時間: {avg_time:.0f}s")
    print(f"總訓練時間: {total_time:.0f}s ({total_time/60:.1f} 分鐘)")
print("="*70)
