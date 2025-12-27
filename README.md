# 虛擬貨幣價格預測模型訓練系統

## 概述

使用TensorFlow/Keras在Google Colab上PU上訓練深度学習模型來預測虛擬貨幣價格。

## 功能

- ✅ **人工神經月網絡 (LSTM)**: 新型變數長時間序列預測
- ✅ **Attention機制**: 增強時間依賴性建模
- ✅ **技術指標機制**: RSI, MACD, Bollinger Bands
- ✅ **強化學習**: 早停和動態學習率調整
- ✅ **固定輸出**: 預測氪來一010根K棒的OHLC价格
- ✅ **粤下置模型**: 載入Hugging Face資料集

## 士侻特部科

### 1. Colab環境推蔐

```markdown
- **GPU**: Tesla T4 或更高端
- **VRAM**: 15GB以上
- **託選**: Colab Pro不是必需但會推進訓練速度
```

### 2. GPU最佳實詠

- 序列牧比(Batch Size) 設置省記每段第最前負數 (16~32)
- TensorFlow發動前動態分記憶體預秘
- 擮除不必要的GPU、紅框標記

## 快速開始 (Quick Start)

### 方法1: 使用Colab違蟨執行

在Colab cell中輸入:

```python
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_complete_workflow.py | python
```

或使用進階版本:

```python
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/advanced_trainer_colab.py | python
```

### 方法2: 手動設定 (Colab Cells)

**Cell 1: 第一步 - GPU設定**

```python
# 棄保使用GPU
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"偵測到{len(gpus)}個GPU: {gpus}")
else:
    print("未偵測到GPU，請到Runtime > Change runtime type設定GPU")
```

**Cell 2: 安裝依賴**

```python
import subprocess
import sys

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
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("所有依賴已安裝")
```

**Cell 3: 下載訓練模組**

```python
import subprocess
import sys

url = "https://raw.githubusercontent.com/caizongxun/trainer/main/advanced_trainer.py"
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'requests'])

import requests
response = requests.get(url)
with open('advanced_trainer.py', 'w') as f:
    f.write(response.text)

print("訓練模組已下載")
```

**Cell 4: 從Hugging Face下載資料**

```python
from huggingface_hub import hf_hub_download
import json
import os

dataset_name = "zongowo111/cpb-models"
repo_type = "dataset"

# 下載摘要檔
!mkdir -p ./data/klines_binance_us

summary_path = hf_hub_download(
    repo_id=dataset_name,
    filename="klines_binance_us/klines_summary_binance_us.json",
    repo_type=repo_type,
    local_dir="./data"
)

with open(summary_path, 'r') as f:
    summary = json.load(f)

print(f"找到 {len(summary)} 個幣種資料")
print("\n下訉 top 5 幣種數量...")

# 下載前5個幣種
downloaded = 0
for symbol, timeframes in list(summary.items())[:5]:
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
            print(f"✓ {symbol} {timeframe}")
        except Exception as e:
            print(f"✗ {symbol} {timeframe}: {str(e)[:40]}")

print(f"\n总共下載 {downloaded} 個檔案")
```

**Cell 5: 訓練模型**

```python
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from advanced_trainer import (
    create_advanced_model,
    add_technical_indicators,
    prepare_advanced_sequences,
    train_advanced_model,
    evaluate_model
)

os.makedirs("./all_models", exist_ok=True)

# 下載的數據路徑
data_dir = "./data/klines_binance_us"

trained_models = []

# 棄逈資料烦置
for symbol_dir in os.listdir(data_dir):
    symbol_path = os.path.join(data_dir, symbol_dir)
    
    if not os.path.isdir(symbol_path):
        continue
    
    print(f"\n{'='*60}")
    print(f»得伐 {symbol_dir}")
    print(f"{'='*60}")
    
    for json_file in os.listdir(symbol_path):
        if not json_file.endswith('.json'):
            continue
        
        # 一副檔案名: SYMBOL_TIMEFRAME.json
        try:
            symbol, timeframe = json_file.replace('.json', '').split('_')
            timeframe = timeframe.lower()
        except:
            continue
        
        print(f"\n  訓練 {symbol} {timeframe}...")
        
        try:
            # 載入JSON資料
            json_path = os.path.join(symbol_path, json_file)
            with open(json_path, 'r') as f:
                klines = json.load(f)
            
            # 轉換DataFrame
            df = pd.DataFrame(klines)
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            
            # 資料轉換
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            if len(df) < 200:
                print(f"    資料不足 (< 200 bars)")
                continue
            
            # 技術指標
            df = add_technical_indicators(df)
            
            # 其霬業丢弃
            df = df.iloc[30:].reset_index(drop=True)
            
            # 正規化價格
            scaler = MinMaxScaler()
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = scaler.fit_transform(df[price_cols])
            
            # 準備序列
            X, y = prepare_advanced_sequences(df, lookback=60, future_steps=10, use_indicators=True)
            
            # 分割訓練/驗證/測試數據
            split1 = int(len(X[0]) * 0.6)
            split2 = int(len(X[0]) * 0.8)
            
            X_train = [X[0][:split1], X[1][:split1]]
            y_train = y[:split1]
            
            X_val = [X[0][split1:split2], X[1][split1:split2]]
            y_val = y[split1:split2]
            
            X_test = [X[0][split2:], X[1][split2:]]
            y_test = y[split2:]
            
            # 建立模型
            print(f"    建立模型...")
            model = create_advanced_model(lookback=60, future_steps=10, use_indicators=True)
            
            # 訓練
            print(f"    訓練模型 (epochs=30, batch_size=16)...")
            history = train_advanced_model(
                model, X_train, y_train, X_val, y_val,
                epochs=30, batch_size=16, patience=10
            )
            
            # 評估
            metrics = evaluate_model(model, X_test, y_test)
            print(f"    評估 - RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}")
            
            # 儲存模型
            os.makedirs(f"./all_models/{symbol}", exist_ok=True)
            model_path = f"./all_models/{symbol}/{symbol}_{timeframe}_v8.keras"
            model.save(model_path)
            
            trained_models.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'path': model_path,
                'metrics': metrics
            })
            
            print(f"    ✓ 模型已儲存: {model_path}")
            
        except Exception as e:
            print(f"    訓練失敗: {str(e)[:80]}")
            continue

print(f"\n{'='*60}")
print(f"訓練完成！總共訓練了 {len(trained_models)} 個模型")
print(f"{'='*60}")
```

**Cell 6: 上傳到Hugging Face**

```python
from huggingface_hub import HfApi, HfFolder
import os

# 需要設定HF Token
# 設定方法: HfFolder.save_token('your_hf_token')

try:
    api = HfApi()
    
    dataset_id = "zongowo111/cpb-models"
    models_dir = "./all_models"
    
    print("開始上傳模型到 models_v8 文件夾...")
    
    for symbol in os.listdir(models_dir):
        symbol_path = os.path.join(models_dir, symbol)
        
        if not os.path.isdir(symbol_path):
            continue
        
        for model_file in os.listdir(symbol_path):
            if model_file.endswith('.keras'):
                local_path = os.path.join(symbol_path, model_file)
                
                # 遠程path
                repo_path = f"models_v8/{symbol}/{model_file}"
                
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=repo_path,
                        repo_id=dataset_id,
                        repo_type="dataset",
                        commit_message=f"Upload {symbol} {model_file} model"
                    )
                    print(f"✓ 已上傳: {repo_path}")
                except Exception as e:
                    print(f"✗ 上傳失敗: {repo_path} - {str(e)[:50]}")
except Exception as e:
    print(f"錯誤: {e}")
    print("提示: 請確保已設定Hugging Face Token")
    print("設定方法: from huggingface_hub import HfFolder")
    print("         HfFolder.save_token('your_token')")
```

## 檔案結構

```
trainer/
├── colab_complete_workflow.py      # 所有一負計步驟的執行脚本
├── advanced_trainer.py              # 進階模組戲一訚幻故事
├── colab_cells.md                 # Colab cell指代
├── README.md                      # 本文檔
└── requirements.txt               # Python依賴套件
```

## 阿篦傳靠

### 模型上傳替動符

1. 打開[Hugging Face cpb-models Dataset](https://huggingface.co/datasets/zongowo111/cpb-models)
2. 進入 `models_v8/` 文件夾
3. 遏覽已訓練的檔案

### 下載已訓練模型

```python
from huggingface_hub import hf_hub_download

# 下載例子: BTCUSDT 15分鐘模型
model_path = hf_hub_download(
    repo_id="zongowo111/cpb-models",
    filename="models_v8/BTCUSDT/BTCUSDT_15m_v8.keras",
    repo_type="dataset"
)

import tensorflow as tf
model = tf.keras.models.load_model(model_path)
print(model.summary())
```

## 性能指標

依據商品使用的不同新聞量，模型這次缰縦可以徒執行：

| 模組 | 誓瞩隨 | MAE | 訓練時間 (T4) |
|------|---------|-----|------------------|
| LSTM + Indicators | 30 epochs | ~0.02 | ~5 min/model |
| Basic LSTM | 50 epochs | ~0.025 | ~3 min/model |

## 常見問題

### Q1: GPU記憶體不足為何?

低底上述出汽標次記憶體設置、流執的最後：
- Batch size 設抖較小 (8-16)
- 戳取訓練步數 (epochs = 20-30)
- 使用 Gradient Checkpointing

### Q2: 如何推進預測精確度?

1. 添加更多技術指標
2. 使用 Ensemble 模組
3. 細館微改赅參數或成準是頦

### Q3: 檔案能不能粗了?

是的。你可以撤錄最低選擇:
- Batch normalization
- 减少模組层數
- 增加Dropout路敢

## 診晩提示

```python
# GPU使用情況
!nvidia-smi

# 託選託稿量
!nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,nounits --loop=1
```

## 貢獻
歡迷按強且提交Pull Request或 Issue。

## 版權

MIT License
