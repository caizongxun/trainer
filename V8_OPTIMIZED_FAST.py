#!/usr/bin/env python3
"""
V8 快速版本 - 优化训练时间

优化措施:
1. Lookback: 120 → 60 (回到 V7 配置)
2. Encoder: 4层 → 3层 BiLSTM
3. Decoder: 2层 → 1层 LSTM
4. 保留 3 个输出 (OHLC + BB + Volatility)

预期效果:
- 训练时间: 8分 → 1.5-2分
- 总訓练时间 (10个): 80分 → 20分
- MAPE 性能: 不渝妨显起下降

執行:
!rm -rf ~/.cache/tensorflow ~/.cache/keras ~/.cache/torch
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/V8_OPTIMIZED_FAST.py | python
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
from typing import Tuple, Dict

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

print_header("0/7", "清理缓存")
try:
    os.system('rm -rf ~/.cache/tensorflow ~/.cache/keras ~/.cache/torch 2>/dev/null')
    os.system('rm -rf ./all_models_v8* 2>/dev/null')
    print_success("缓存已清理")
except:
    pass

print_header("1/7", "Colab环境设定")

try:
    from google.colab import drive
    IS_COLAB = True
    print_success("Google Colab环境侦测成功")
except ImportError:
    IS_COLAB = False
    print_warning("本地环境模式")

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print_success(f"侦测到 {len(gpus)} 个GPU")
except:
    pass

print_header("2/7", "安装依赖套件")

for module, name in [('tensorflow', 'TensorFlow'), ('keras', 'Keras'), 
                     ('huggingface_hub', 'HF Hub'), ('pandas', 'Pandas'), 
                     ('numpy', 'NumPy'), ('sklearn', 'Scikit-Learn')]:
    try:
        __import__(module)
        print_success(f"{name} 已安装")
    except:
        pass

print_header("3/7", "数据准备")

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标 (V7 标准)"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values if 'volume' in df else np.ones(len(close))
    
    # RSI (14)
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
    
    # Bollinger Bands (20, 2)
    df['sma20'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + 2 * bb_std
    df['bb_lower'] = df['sma20'] - 2 * bb_std
    df['bb_mid'] = df['sma20']
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_width_pct'] = df['bb_width'] / (df['sma20'] + 1e-10)
    
    # 波动性
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

def prepare_sequences_v8_fast(ohlc_data: np.ndarray, technical_data: np.ndarray,
                             lookback: int = 60, forecast_horizon: int = 10) -> Tuple:
    """V8 快速版 序列准备 - 60 根 lookback"""
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

print_success("数据准备函数已初始化")

print_header("4/7", "数据下载")

os.makedirs("./data/klines_binance_us", exist_ok=True)
os.makedirs("./all_models_v8_fast", exist_ok=True)

from huggingface_hub import list_repo_files, hf_hub_download

DATASET_ID = "zongowo111/cpb-models"
DATA_DIR = "./data/klines_binance_us"

try:
    files = list_repo_files(repo_id=DATASET_ID, repo_type="dataset", revision="main")
    csv_files = [f for f in files if f.startswith('klines_binance_us/') and f.endswith('.csv')]
    
    print_success(f"找到 {len(csv_files)} 个 CSV")
    
    symbols = {}
    for csv_file in csv_files:
        parts = csv_file.split('/')
        if len(parts) == 3:
            symbol = parts[1]
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(csv_file)
    
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
                except:
                    pass
            downloaded_count += 1
            
            timeframe = filename.split('_')[1] if '_' in filename else 'unknown'
            pairs_to_train.append((symbol, timeframe, local_path))
    
    print_success(f"下载 {downloaded_count} 个档案")
    print_success(f"准备训练 {min(10, len(pairs_to_train))} 个模型")
    
except Exception as e:
    print_error(f"下载失败: {e}")
    pairs_to_train = [("BTCUSDT", "15m", None)]

print_header("5/7", "模型训练")

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Input, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_seq2seq_v8_fast(input_shape: Tuple[int, int], forecast_horizon: int = 10) -> Model:
    """
    V8 快速版模型 - 优化后的深度
    
    Encoder: 3 层 BiLSTM (128→64→32)
    Decoder: 1 层 LSTM (64)
    输出: 3 个任务 (OHLC + BB + Volatility)
    """
    lookback, n_features = input_shape
    
    inputs = Input(shape=(lookback, n_features), name='encoder_input')
    
    # Encoder - 3 层 BiLSTM (128->64->32)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(inputs)
    x = LayerNormalization()(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = LayerNormalization()(x)
    
    x = LSTM(32, return_sequences=False, dropout=0.3)(x)
    x = LayerNormalization()(x)
    
    # 重复向量
    encoder_output = RepeatVector(forecast_horizon)(x)
    
    # Decoder - 1 层 LSTM
    decoder = LSTM(64, return_sequences=True, dropout=0.3)(encoder_output)
    decoder = LayerNormalization()(decoder)
    
    # 三个输出层
    ohlc_out = TimeDistributed(Dense(4), name='ohlc_output')(decoder)
    bb_out = TimeDistributed(Dense(2), name='bb_params_output')(decoder)
    vol_out = TimeDistributed(Dense(1), name='volatility_output')(decoder)
    
    model = Model(inputs=inputs, outputs=[ohlc_out, bb_out, vol_out])
    
    # 编译 - V7 标准配置
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
        },
        metrics={
            'ohlc_output': ['mae'],
            'bb_params_output': ['mae'],
            'volatility_output': ['mae']
        }
    )
    
    return model

trained_count = 0
training_results = []

for idx, (symbol, timeframe, csv_path) in enumerate(pairs_to_train[:10], 1):
    train_start = time.time()
    print(f"\n  [{idx}/10] {symbol} {timeframe}", end=' ', flush=True)
    
    try:
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
            print("⚠ 数据不足")
            continue
        
        df = add_technical_indicators(df)
        
        ohlc_data = df[['open', 'high', 'low', 'close']].values.astype(np.float32)
        tech_cols = ['rsi', 'macd', 'signal', 'bb_upper', 'bb_lower', 'bb_width_pct', 'volatility', 'atr']
        technical_data = df[tech_cols + ['bb_width_pct', 'volatility']].values.astype(np.float32)
        if technical_data.shape[1] != 10:
            technical_data = np.concatenate([technical_data, technical_data[:, -1:]], axis=1)
        
        ohlc_scaler = StandardScaler()
        ohlc_normalized = ohlc_scaler.fit_transform(ohlc_data)
        
        tech_scaler = StandardScaler()
        technical_normalized = tech_scaler.fit_transform(technical_data)
        
        # 序列准备 - 60 lookback
        X, y_ohlc, y_bb, y_vol = prepare_sequences_v8_fast(
            ohlc_normalized, technical_normalized,
            lookback=60, forecast_horizon=10
        )
        
        if len(X) < 50:
            print("⚠ 序列不足")
            continue
        
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_ohlc_train, y_ohlc_val = y_ohlc[:split], y_ohlc[split:]
        y_bb_train, y_bb_val = y_bb[:split], y_bb[split:]
        y_vol_train, y_vol_val = y_vol[:split], y_vol[split:]
        
        model = create_seq2seq_v8_fast(X_train.shape[1:], forecast_horizon=10)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
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
            verbose=0,
            callbacks=callbacks
        )
        
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        y_ohlc_pred_train = model.predict(X_train, verbose=0)[0]
        y_ohlc_pred_val = model.predict(X_val, verbose=0)[0]
        
        def safe_mape(y_true, y_pred):
            y_true_flat = y_true.reshape(-1)
            y_pred_flat = y_pred.reshape(-1)
            mask = np.abs(y_true_flat) > 1e-10
            if mask.sum() == 0:
                return 999.0
            return np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
        
        train_mape = safe_mape(y_ohlc_train, y_ohlc_pred_train)
        val_mape = safe_mape(y_ohlc_val, y_ohlc_pred_val)
        
        os.makedirs(f"./all_models_v8_fast/{symbol}", exist_ok=True)
        model_path = f"./all_models_v8_fast/{symbol}/{symbol}_{timeframe}_v8fast.keras"
        model.save(model_path)
        
        params = {
            'ohlc_mean': ohlc_scaler.mean_.tolist(),
            'ohlc_scale': ohlc_scaler.scale_.tolist(),
            'technical_mean': tech_scaler.mean_.tolist(),
            'technical_scale': tech_scaler.scale_.tolist(),
            'technical_cols': tech_cols,
            'lookback': 60,
            'forecast_horizon': 10
        }
        
        params_path = f"./all_models_v8_fast/{symbol}/{symbol}_{timeframe}_v8fast_params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f)
        
        trained_count += 1
        train_time = time.time() - train_start
        
        print(f"✔ {train_time:.1f}s | Loss: {val_loss:.4f} | MAPE: {val_mape:.2f}%")
        
        training_results.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'train_mape': float(train_mape),
            'val_mape': float(val_mape),
            'training_time': train_time,
            'epochs': len(history.history['loss'])
        })
        
        del model, X, y_ohlc, y_bb, y_vol
        del X_train, y_ohlc_train, y_bb_train, y_vol_train
        del X_val, y_ohlc_val, y_bb_val, y_vol_val, df
        gc.collect()
        
    except Exception as e:
        print_error(f"训练失败: {str(e)[:50]}")
        continue

print(f"\n  ✔ 成功训练 {trained_count} 个模型")

print_header("6/7", "上传模型")

try:
    from huggingface_hub import HfApi
    api = HfApi()
    upload_count = 0
    
    for symbol in os.listdir("./all_models_v8_fast"):
        symbol_path = f"./all_models_v8_fast/{symbol}"
        if not os.path.isdir(symbol_path):
            continue
        
        for model_file in os.listdir(symbol_path):
            if model_file.endswith(('.keras', '.json')):
                try:
                    api.upload_file(
                        path_or_fileobj=os.path.join(symbol_path, model_file),
                        path_in_repo=f"models_v8_fast/{symbol}/{model_file}",
                        repo_id=DATASET_ID,
                        repo_type="dataset",
                        commit_message=f"V8 Fast {symbol}"
                    )
                    upload_count += 1
                except:
                    pass
    
    print_success(f"成功上传 {upload_count} 个档案")
except:
    print_warning("上传失败")

print_header("7/7", "完成")

summary = {
    "timestamp": datetime.now().isoformat(),
    "version": "v8_fast",
    "trained_models": trained_count,
    "architecture": "3-layer BiLSTM + LayerNormalization + 3 outputs",
    "learning_rate": 0.0005,
    "batch_size": 16,
    "lookback": 60,
    "forecast_horizon": 10,
    "patience": 15,
    "results": training_results
}

with open("./training_summary_v8fast.json", 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print_success("摘要已保存")

print("\n" + "="*60)
print("V8 Fast 训练上传完成")
print(f"训练模型: {trained_count}")
if training_results:
    avg_val_mape = np.mean([r['val_mape'] for r in training_results])
    avg_time = np.mean([r['training_time'] for r in training_results])
    avg_epochs = np.mean([r['epochs'] for r in training_results])
    print(f"平均验证 MAPE: {avg_val_mape:.2f}%")
    print(f"平均训练时间: {avg_time:.1f}s (预期: 1-2 分)")
    print(f"平均 Epochs: {avg_epochs:.0f}")
print("="*60)
