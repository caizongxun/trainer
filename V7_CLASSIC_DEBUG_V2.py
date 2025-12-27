#!/usr/bin/env python3
"""
V7 Classic Debug V2 - 即時記憶體監控，檢查是否真的在訓練

主要改進:
✅ 每一 epoch 即時輸出（不是後接轉入）
✅ 詳細的記憶體監控 (RAM, GPU RAM)
✅ 士貼從 GPU 和 CPU 的架構檔讁殾華蝶
✅ 訓練 timeout 検測（如果不動的了）
✅ 逩子詐文戲流（解決闖始殊止輸出。

執行:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/V7_CLASSIC_DEBUG_V2.py | python
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
import psutil
import threading
from typing import Tuple

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_memory_info():
    """取得系統和 GPU 記憶體使用情況"""
    # 系統記憶體
    mem = psutil.virtual_memory()
    sys_ram_used = mem.used / (1024**3)
    sys_ram_total = mem.total / (1024**3)
    
    # GPU 記憶體 (簡單估計)
    gpu_info = ""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # 獲取 GPU 記憶體使用
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            gpu_used = gpu_memory['current'] / (1024**3) if 'current' in gpu_memory else 0
            gpu_total = gpu_memory['peak'] / (1024**3) if 'peak' in gpu_memory else 15.0
        else:
            gpu_used, gpu_total = 0, 0
    except:
        gpu_used, gpu_total = 0, 0
    
    return sys_ram_used, sys_ram_total, gpu_used, gpu_total

def debug_print(message, level="INFO"):
    """帶時間戳的 Debug 輸出"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if level == "INFO":
        print(f"[{timestamp}] [INFO] {message}")
    elif level == "WARN":
        print(f"[{timestamp}] [WARN] {message}")
    elif level == "ERROR":
        print(f"[{timestamp}] [ERROR] {message}")
    elif level == "DEBUG":
        print(f"[{timestamp}] [DEBUG] {message}")
    elif level == "SUCCESS":
        print(f"[{timestamp}] [✓] {message}")
    elif level == "MEMORY":
        sys_used, sys_total, gpu_used, gpu_total = get_memory_info()
        print(f"[{timestamp}] [MEM] 系統: {sys_used:.1f}/{sys_total:.1f}GB | GPU: {gpu_used:.1f}/{gpu_total:.1f}GB")
    sys.stdout.flush()

def print_header(step, message):
    print(f"\n[{step}/7] {message}")
    sys.stdout.flush()

print_header("0/7", "初始化檢查")

debug_print("開始執行 V7 Classic Debug V2 版本")
debug_print(f"Python 版本: {sys.version}")
debug_print(f"當前工作目錄: {os.getcwd()}")

# 檢查是否在 Colab
try:
    from google.colab import drive
    IS_COLAB = True
    debug_print("偵測到 Google Colab 環境", "SUCCESS")
except ImportError:
    IS_COLAB = False
    debug_print("本地環境模式", "WARN")

print_header("1/7", "TensorFlow 初始化 (詳細診斷)")

debug_print("導入 TensorFlow...")
try:
    import tensorflow as tf
    debug_print(f"TensorFlow 版本: {tf.__version__}", "SUCCESS")
except Exception as e:
    debug_print(f"TensorFlow 導入失敗: {e}", "ERROR")
    sys.exit(1)

# GPU 診斷
debug_print("檢查 GPU 設備...")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        debug_print(f"偵測到 {len(gpus)} 個物理 GPU", "SUCCESS")
        for i, gpu in enumerate(gpus):
            debug_print(f"  GPU {i}: {gpu.name}")
        
        # 設定 GPU 記憶體動態增長
        debug_print("設定 GPU 記憶體動態增長...")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                debug_print(f"  {gpu.name}: 已啟用動態增長", "SUCCESS")
            except Exception as e:
                debug_print(f"  {gpu.name}: 設定失敗 - {e}", "WARN")
    else:
        debug_print("未偵測到 GPU (使用 CPU 模式)", "WARN")
except Exception as e:
    debug_print(f"GPU 檢查失敗: {e}", "ERROR")

print_header("2/7", "安裝依賴套件")

for module, name in [('tensorflow', 'TensorFlow'), ('keras', 'Keras'), 
                     ('huggingface_hub', 'HF Hub'), ('pandas', 'Pandas'), 
                     ('numpy', 'NumPy'), ('sklearn', 'Scikit-Learn')]:
    try:
        __import__(module)
        debug_print(f"{name} 已安裝", "SUCCESS")
    except Exception as e:
        debug_print(f"{name} 導入失敗: {e}", "ERROR")

print_header("3/7", "數據準備函數")

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算技術指標 (V7 標準)"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    debug_print(f"計算技術指標 (資料點數: {len(df)})", "DEBUG")
    
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
    debug_print("  ✓ RSI 計算完成", "DEBUG")
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    debug_print("  ✓ MACD 計算完成", "DEBUG")
    
    # Bollinger Bands (20, 2)
    df['sma20'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + 2 * bb_std
    df['bb_lower'] = df['sma20'] - 2 * bb_std
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_width_pct'] = df['bb_width'] / (df['sma20'] + 1e-10)
    debug_print("  ✓ Bollinger Bands 計算完成", "DEBUG")
    
    # 波動性
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
    debug_print("  ✓ ATR 計算完成", "DEBUG")
    
    result = df.fillna(method='bfill').fillna(method='ffill')
    debug_print(f"技術指標計算完成 (結果行數: {len(result)})", "SUCCESS")
    return result

def prepare_sequences(ohlc_data: np.ndarray, technical_data: np.ndarray,
                     lookback: int = 60, forecast_horizon: int = 10) -> Tuple:
    """V7 序列準備"""
    debug_print(f"準備序列 (lookback={lookback}, horizon={forecast_horizon})", "DEBUG")
    
    X, y_ohlc, y_bb_params, y_volatility = [], [], [], []
    
    for i in range(len(ohlc_data) - lookback - forecast_horizon):
        X.append(np.concatenate([
            ohlc_data[i:i+lookback],
            technical_data[i:i+lookback]
        ], axis=1))
        
        y_ohlc.append(ohlc_data[i+lookback:i+lookback+forecast_horizon])
        y_bb_params.append(technical_data[i+lookback:i+lookback+forecast_horizon, 4:6])
        y_volatility.append(technical_data[i+lookback:i+lookback+forecast_horizon, 8:9])
    
    X_array = np.array(X)
    y_ohlc_array = np.array(y_ohlc)
    y_bb_array = np.array(y_bb_params)
    y_vol_array = np.array(y_volatility)
    
    debug_print(f"序列準備完成: X {X_array.shape}", "SUCCESS")
    return X_array, y_ohlc_array, y_bb_array, y_vol_array

debug_print("數據準備函數已初始化", "SUCCESS")

print_header("4/7", "數據下載 (詳細日誌)")

os.makedirs("./data/klines_binance_us", exist_ok=True)
os.makedirs("./all_models_v7_debug_v2", exist_ok=True)

from huggingface_hub import list_repo_files, hf_hub_download

DATASET_ID = "zongowo111/cpb-models"
DATA_DIR = "./data/klines_binance_us"

debug_print(f"HF 資料集 ID: {DATASET_ID}")

try:
    debug_print("列出 HF 資料集檔案...")
    files = list_repo_files(repo_id=DATASET_ID, repo_type="dataset", revision="main")
    debug_print(f"總共 {len(files)} 個檔案", "DEBUG")
    
    csv_files = [f for f in files if f.startswith('klines_binance_us/') and f.endswith('.csv')]
    debug_print(f"找到 {len(csv_files)} 個 CSV 檔案", "SUCCESS")
    
    # 組織數據按幣種
    symbols = {}
    for csv_file in csv_files:
        parts = csv_file.split('/')
        if len(parts) == 3:
            symbol = parts[1]
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(csv_file)
    
    debug_print(f"找到 {len(symbols)} 個幣種:", "SUCCESS")
    pairs_to_train = []
    
    for symbol in sorted(symbols.keys()):
        symbol_path = f"{DATA_DIR}/{symbol}"
        os.makedirs(symbol_path, exist_ok=True)
        
        for csv_file in symbols[symbol]:
            filename = csv_file.split('/')[-1]
            local_path = f"{symbol_path}/{filename}"
            
            if not os.path.exists(local_path):
                debug_print(f"下載 {filename}...", "DEBUG")
                try:
                    hf_hub_download(repo_id=DATASET_ID, filename=csv_file,
                                  repo_type="dataset", local_dir="./data")
                    debug_print(f"  ✓ {filename} 下載成功", "DEBUG")
                except Exception as e:
                    debug_print(f"  ✗ {filename} 下載失敗: {e}", "WARN")
            else:
                debug_print(f"{filename} 已存在，跳過", "DEBUG")
            
            timeframe = filename.split('_')[1] if '_' in filename else 'unknown'
            pairs_to_train.append((symbol, timeframe, local_path))
    
    debug_print(f"下載完成 (準備訓練 {min(40, len(pairs_to_train))} 個模型)", "SUCCESS")
    
except Exception as e:
    debug_print(f"下載失敗: {e}", "ERROR")
    pairs_to_train = [("BTCUSDT", "15m", None)]

print_header("5/7", "模型訓練 (即時監控)")

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Input, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

class DebugCallback(Callback):
    """即時監控訓練進度和記憶體"""
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        loss = logs.get('loss', 0) if logs else 0
        val_loss = logs.get('val_loss', 0) if logs else 0
        
        # 每 5 個 epoch 輸出一次
        if epoch % 5 == 0 or epoch < 3:
            sys_used, sys_total, gpu_used, gpu_total = get_memory_info()
            debug_print(
                f"Epoch {epoch:3d}: loss={loss:.6f} | val_loss={val_loss:.6f} | "
                f"RAM {sys_used:.1f}GB | GPU {gpu_used:.1f}GB | {epoch_time:.1f}s",
                "DEBUG"
            )

def create_seq2seq_v7(input_shape: Tuple[int, int], forecast_horizon: int = 10) -> Model:
    """V7 模型"""
    lookback, n_features = input_shape
    debug_print(f"建立 V7 模型 (input_shape={input_shape})", "DEBUG")
    
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
    
    debug_print("模型編譯中...", "DEBUG")
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
    
    debug_print("模型編譯完成", "SUCCESS")
    return model

trained_count = 0
training_results = []
max_pairs = min(5, len(pairs_to_train))  # 只訓練 5 個以快速測試

for idx, (symbol, timeframe, csv_path) in enumerate(pairs_to_train[:max_pairs], 1):
    train_start = time.time()
    print(f"\n[{idx}/{max_pairs}] {symbol} {timeframe}", end='', flush=True)
    debug_print(f"\n開始訓練 [{idx}/{max_pairs}] {symbol} {timeframe}")
    
    try:
        # 讀取數據
        debug_print("讀取數據檔案...")
        if csv_path and os.path.exists(csv_path):
            debug_print(f"  路徑: {csv_path}", "DEBUG")
            df = pd.read_csv(csv_path)
            debug_print(f"  檔案大小: {df.shape}", "DEBUG")
            df.columns = [col.lower().strip() for col in df.columns]
        else:
            debug_print(f"檔案不存在，生成模擬數據", "WARN")
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
            debug_print(f"數據不足 ({len(df)} < 500)", "WARN")
            print(" ⚠ 數據不足")
            continue
        
        # 計算技術指標
        debug_print("計算技術指標...")
        df = add_technical_indicators(df)
        
        # 準備數據
        debug_print("準備 OHLC 數據...")
        ohlc_data = df[['open', 'high', 'low', 'close']].values.astype(np.float32)
        tech_cols = ['rsi', 'macd', 'signal', 'bb_upper', 'bb_lower', 'bb_width_pct', 'volatility', 'atr']
        technical_data = df[tech_cols + ['bb_width_pct', 'volatility']].values.astype(np.float32)
        
        if technical_data.shape[1] != 10:
            technical_data = np.concatenate([technical_data, technical_data[:, -1:]], axis=1)
        
        # 標準化
        debug_print("數據標準化...")
        ohlc_scaler = StandardScaler()
        ohlc_normalized = ohlc_scaler.fit_transform(ohlc_data)
        debug_print("  OHLC 標準化完成", "DEBUG")
        
        tech_scaler = StandardScaler()
        technical_normalized = tech_scaler.fit_transform(technical_data)
        debug_print("  技術資料標準化完成", "DEBUG")
        
        # 序列準備
        debug_print("準備序列...")
        X, y_ohlc, y_bb, y_vol = prepare_sequences(
            ohlc_normalized, technical_normalized,
            lookback=60, forecast_horizon=10
        )
        
        if len(X) < 50:
            debug_print(f"序列數量不足 ({len(X)} < 50)", "WARN")
            print(" ⚠ 序列不足")
            continue
        
        # 分割訓練/驗證集
        debug_print("分割訓練/驗證集 (80/20)...")
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_ohlc_train, y_ohlc_val = y_ohlc[:split], y_ohlc[split:]
        y_bb_train, y_bb_val = y_bb[:split], y_bb[split:]
        y_vol_train, y_vol_val = y_vol[:split], y_vol[split:]
        
        debug_print(f"訓練集大小: {X_train.shape[0]}", "DEBUG")
        debug_print(f"驗證集大小: {X_val.shape[0]}", "DEBUG")
        
        # 建立模型
        debug_print("建立模型...")
        model = create_seq2seq_v7(X_train.shape[1:], forecast_horizon=10)
        
        # 訓練
        debug_print("開始訓練...", "DEBUG")
        debug_print("  Epochs: 100", "DEBUG")
        debug_print("  Batch Size: 16", "DEBUG")
        debug_print("  訓練樣本數: " + str(X_train.shape[0]), "DEBUG")
        debug_print("  ↓ 即將進入訓練迴圈", "INFO")
        debug_print("", "MEMORY")  # 訓練前記憶體快照
        
        print(f" → 訓練中...", end='', flush=True)
        sys.stdout.flush()
        
        training_start = time.time()
        
        callbacks = [
            DebugCallback(),  # 自訂 callback，即時輸出進度
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        # 訓練
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
            verbose=0,  # 不使用 Keras 預設輸出
            callbacks=callbacks
        )
        
        training_time = time.time() - training_start
        debug_print("", "MEMORY")  # 訓練後記憶體快照
        debug_print(f"訓練完成 (耗時 {training_time:.2f}s)", "SUCCESS")
        
        # 評估
        debug_print("進行預測...", "DEBUG")
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
        debug_print("保存模型...", "DEBUG")
        os.makedirs(f"./all_models_v7_debug_v2/{symbol}", exist_ok=True)
        model_path = f"./all_models_v7_debug_v2/{symbol}/{symbol}_{timeframe}_v7.keras"
        model.save(model_path)
        debug_print(f"  模型保存至: {model_path}", "SUCCESS")
        
        # 保存參數
        params = {
            'ohlc_mean': ohlc_scaler.mean_.tolist(),
            'ohlc_scale': ohlc_scaler.scale_.tolist(),
            'technical_mean': tech_scaler.mean_.tolist(),
            'technical_scale': tech_scaler.scale_.tolist(),
            'version': 'v7_classic_debug_v2'
        }
        
        params_path = f"./all_models_v7_debug_v2/{symbol}/{symbol}_{timeframe}_v7_params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f)
        debug_print(f"  參數保存至: {params_path}", "SUCCESS")
        
        trained_count += 1
        train_time = time.time() - train_start
        
        print(f" ✓ {train_time:.1f}s | Loss: {history.history['val_loss'][-1]:.4f} | MAPE: {val_mape:.2f}%")
        debug_print(f"訓練成功 (總耗時 {train_time:.2f}s)", "SUCCESS")
        
        training_results.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'val_loss': float(history.history['val_loss'][-1]),
            'val_mape': float(val_mape),
            'training_time': train_time,
            'epochs': len(history.history['loss'])
        })
        
        # 釋放記憶體
        debug_print("釋放記憶體...", "DEBUG")
        del model, X, y_ohlc, y_bb, y_vol
        del X_train, y_ohlc_train, y_bb_train, y_vol_train
        del X_val, y_ohlc_val, y_bb_val, y_vol_val, df
        gc.collect()
        debug_print("記憶體已釋放", "SUCCESS")
        
    except Exception as e:
        print(f" ✗ 失敗")
        debug_print(f"訓練失敗: {str(e)}", "ERROR")
        import traceback
        debug_print(traceback.format_exc(), "DEBUG")
        continue

print(f"\n  ✓ 成功訓練 {trained_count} 個模型")
debug_print(f"訓練階段完成 (成功: {trained_count}/{max_pairs})", "SUCCESS")

print_header("6/7", "上傳模型")
debug_print("模型保存在 ./all_models_v7_debug_v2 目錄", "INFO")

print_header("7/7", "完成")

summary = {
    "timestamp": datetime.now().isoformat(),
    "version": "v7_classic_debug_v2",
    "trained_models": trained_count,
    "results": training_results
}

with open("./training_summary_v7_debug_v2.json", 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

debug_print("摘要已保存", "SUCCESS")

print("\n" + "="*60)
print("V7 Classic Debug V2 訓練完成")
print(f"訓練模型: {trained_count}")
if training_results:
    avg_val_mape = np.mean([r['val_mape'] for r in training_results])
    avg_time = np.mean([r['training_time'] for r in training_results])
    print(f"平均驗證 MAPE: {avg_val_mape:.2f}%")
    print(f"平均訓練時間: {avg_time:.1f}s")
print("="*60)

debug_print(f"\n執行完全結束", "SUCCESS")
