# Colab 執行指南

## 方法一： 一步完成 (推荐)

在Google Colab的一個新的cell中，輸入此指令:

```python
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/advanced_trainer_colab.py | python
```
然後按下 `Shift + Enter` 執行。
伺何單粕計箱會自動完成整個訓練流程，搎修正訓練後的檔案會被上傳到HuggingFace dataset 的 `models_v8` 文件夾。

---

## 方法二： 手動設定 (Cell by Cell)

如果你思磨教叫驟地介滌每一步驟骣，下方下諭按序準備推動蜃實。

### Cell 1: 環境設定與GPU优化

輸入此cell法技工：

```python
# GPU檢查與設定動態記憶體分配
import tensorflow as tf

print("="*60)
print("棄保 GPU 情況")
print("="*60)

# 棄保存在的GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"棄俟到 {len(gpus)} 個 GPU")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    # 動態分配 GPU 記憶體
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("\n  正物宐 GPU 記憶體動態分配...")
else:
    print("⚠ 未偵測到 GPU")
    print("  請到 'Runtime' > 'Change runtime type'")
    print("  徜世紀一下 'GPU' 來情場")

# 棄俟 CUDA 和 cuDNN 版本
 print(f"\n  TensorFlow 版本: {tf.__version__}")

print("\n" + "="*60 + "\n")
```

### Cell 2: 安裝依賴套件

```python
import subprocess
import sys

print("="*60)
print("安裝依賴套件")
print("="*60)

packages = [
    'tensorflow>=2.13.0',
    'keras>=2.13.0',
    'huggingface-hub>=0.17.0',
    'pandas',
    'numpy',
    'scikit-learn',
    'requests'
]

for package in packages:
    try:
        module_name = package.split('>=')[0].split('<')[0].replace('-', '_')
        __import__(module_name)
        print(f"  ✔ {package}")
    except ImportError:
        print(f"  正在安裝 {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("\n  所有依賴已安裝")
print("\n" + "="*60 + "\n")
```

### Cell 3: 下載訓練模組及訓練脚本

```python
import requests

print("="*60)
print("下載訓練模組")
print("="*60)

# 下載 advanced_trainer.py
url = "https://raw.githubusercontent.com/caizongxun/trainer/main/advanced_trainer.py"
response = requests.get(url)
with open('advanced_trainer.py', 'w') as f:
    f.write(response.text)
print("  ✔ advanced_trainer.py")

# 下載 colab_trainer.py
url = "https://raw.githubusercontent.com/caizongxun/trainer/main/advanced_trainer_colab.py"
response = requests.get(url)
with open('colab_trainer.py', 'w') as f:
    f.write(response.text)
print("  ✔ colab_trainer.py")

print("\n" + "="*60 + "\n")
```

### Cell 4: 從Hugging Face下載K線資料

```python
from huggingface_hub import hf_hub_download
import json
import os

print("="*60)
print("資料下載")
print("="*60)

dataset_name = "zongowo111/cpb-models"
repo_type = "dataset"

# 創建目錄
!mkdir -p ./data/klines_binance_us

# 下載摘要檔
print("\n  下載戈胡檔...")
summary_path = hf_hub_download(
    repo_id=dataset_name,
    filename="klines_binance_us/klines_summary_binance_us.json",
    repo_type=repo_type,
    local_dir="./data"
)

with open(summary_path, 'r') as f:
    summary = json.load(f)

print(f"  ✔ 找到 {len(summary)} 個幣種")

# 下載中測量的數據 (需要第一步)
print("\n  下載前 10 個幣種的數據...")
downloaded = 0
for symbol, timeframes in list(summary.items())[:10]:
    for timeframe in timeframes:
        filename = f"klines_binance_us/{symbol}/{symbol}_{timeframe}.json"
        try:
            hf_hub_download(
                repo_id=dataset_name,
                filename=filename,
                repo_type=repo_type,
                local_dir="./data"
            )
            downloaded += 1
            print(f"    ✔ {symbol} {timeframe}")
        except Exception as e:
            pass

print(f"\n  成功下載 {downloaded} 個檔案")
print("\n" + "="*60 + "\n")
```

### Cell 5: 訓練模型

上次下載的每個幣種訓練模型：

```python
# 此 cell 是全、惦型的 - 需要徆先成 colab_trainer.py 然後徜下一個 cell 從後执行
def run_training():
    exec(open('colab_trainer.py').read())

run_training()
```
或次了知道的：

```python
# 直接徜下段 cell 一個一個的執行
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import MinMaxScaler
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import gc

# 技術指標凖讀
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

# 粗鮯資料流程
dataset_name = "zongowo111/cpb-models"
data_dir = "./data/klines_binance_us"
os.makedirs("./all_models", exist_ok=True)

trained = 0

for symbol_dir in sorted(os.listdir(data_dir)):
    symbol_path = os.path.join(data_dir, symbol_dir)
    
    if not os.path.isdir(symbol_path):
        continue
    
    print(f"\n{symbol_dir}")
    print("="*40)
    
    for json_file in os.listdir(symbol_path):
        if not json_file.endswith('.json'):
            continue
        
        try:
            # 能慢驅方椿光
            symbol, timeframe = json_file.replace('.json', '').split('_')
            timeframe = timeframe.lower()
        except:
            continue
        
        print(f"{symbol} {timeframe}...", end=' ')
        
        try:
            # 載入資料
            json_path = os.path.join(symbol_path, json_file)
            with open(json_path, 'r') as f:
                klines = json.load(f)
            
            df = pd.DataFrame(klines)
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            if len(df) < 200:
                print("skip")
                continue
            
            # 技術指標
            df = add_technical_indicators(df)
            df = df.iloc[30:].reset_index(drop=True)
            
            # 正規化
            scaler = MinMaxScaler()
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = scaler.fit_transform(df[price_cols])
            
            # 準備序列
            data = df[['open', 'high', 'low', 'close', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position']].values
            
            X, y = [], []
            for i in range(len(data) - 60 - 10 + 1):
                X.append(data[i:i+60])
                y.append(data[i+60:i+60+10, :4].flatten())
            
            X = np.array(X)
            y = np.array(y)
            
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]
            
            # 模型
            model = Sequential([
                LSTM(128, activation='relu', input_shape=(60, 9), return_sequences=True),
                Dropout(0.2),
                LSTM(64, activation='relu', return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(40)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # 訓練
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=16, verbose=0)
            
            # 儲存
            os.makedirs(f"./all_models/{symbol}", exist_ok=True)
            model_path = f"./all_models/{symbol}/{symbol}_{timeframe}_v8.keras"
            model.save(model_path)
            
            trained += 1
            print("ok")
            
            del model, X, y, X_train, y_train, X_val, y_val, df
            gc.collect()
            
        except Exception as e:
            print(f"fail - {str(e)[:30]}")

print(f"\n\n成功訓練: {trained}")
```

### Cell 6: 上傳到Hugging Face

```python
from huggingface_hub import HfApi
import os

print("="*60)
print("上傳模型到Hugging Face")
print("="*60)

try:
    api = HfApi()
    dataset_id = "zongowo111/cpb-models"
    models_dir = "./all_models"
    
    upload_count = 0
    
    for symbol in os.listdir(models_dir):
        symbol_path = os.path.join(models_dir, symbol)
        
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
                        repo_id=dataset_id,
                        repo_type="dataset",
                        commit_message=f"Upload {symbol} model v8"
                    )
                    print(f"  ✔ {repo_path}")
                    upload_count += 1
                except Exception as e:
                    print(f"  ⚠ {repo_path}: {str(e)[:40]}")
    
    print(f"\n  成功上傳: {upload_count}")
    
except Exception as e:
    print(f"  ⚠ 錯誤: {e}")
    print("\n  知譧: HF 需要驅權")

print("\n" + "="*60)
```

---

## 法帯提示

### GPU 紁例憶體不足

1. 減少 batch size (8 來 16)
2. 減少訓練 epochs (15 來 25)
3. 減少 model 有嚣量
4. 使用梧业梧梧梧梧 (Gradient Checkpointing)

### 強化學習

1. 增加 Dropout 率
2. 添加 L1/L2 正規化
3. 使用 Early Stopping
4. 使用學習率調整

### 新新遞代

你能進一步改進 `advanced_trainer.py` 檔案，例如：

- 添加彈成床模型 (Ensemble)
- 使用我寶寶技術指標
- 寶寶準需輸出檔案 (例如 {symbol}_parameters.json)

---

## 上傳檔案二次窡執

屬二徜訓練後需要上傳无光的檔案，你需要：

1. 紦副 Python 檔案 (advanced_trainer.py, colab_complete_workflow.py) 上傳到 trainer repo
2. 紦副訓練完成的檔案上傳到 cpb-models dataset 的 models_v8 文件夾

---

## 文章參考

- 更多 Colab 技巧: https://github.com/caizongxun/trainer
- TensorFlow GPU 檢查: https://www.tensorflow.org/guide/gpu
- LSTM 時間序列: https://www.tensorflow.org/guide/keras/rnn
