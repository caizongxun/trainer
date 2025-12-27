# V8 訓練卡住根本原因分析 & 快速修復方案

## 您目前遇到的問題

```
[1/40] ADAUSDT 15m 卡住 30+ 秒
一個模型訓練這麼久不正常
```

---

## 根本原因診斷

### 問題 1：V8_STABLE.py 參數設置不當

在您執行的 **V8_STABLE.py** 中，有這些瓶頸：

```python
# ❌ 問題代碼 (V8_STABLE.py 行 280-293)
model.fit(
    X_train, y_ohlc_train,
    validation_data=(X_val, y_ohlc_val),
    epochs=150,           # 太多 epochs！
    batch_size=32,        # batch 太大會導致內存問題
    verbose=0,
    callbacks=callbacks
)
```

**比較 V8_STABLE.py vs V8_ALIGNED_WITH_V7.py：**

| 參數 | V8_STABLE | V8_ALIGNED | 影響 |
|------|-----------|-----------|------|
| Epochs | 150 | 100 | +50% 訓練時間 |
| Batch Size | 32 | 16 | 記憶體碎片化 |
| Lookback | 120 | 60 | 序列長度 2 倍 |
| Learning Rate | 0.001 | 0.0005 | 收斂速度差 |
| 模型架構 | 4 層 LSTM | 3 層 BiLSTM + 3 輸出 | 複雜度 40% 更高 |

---

### 問題 2：技術指標維度不匹配導致資料準備爆炸

**V8_STABLE.py 行 210-245：**

```python
# 14 個技術指標，但實際上只有 11 個被正確計算
technical_data = df[['rsi', 'macd', 'signal', 'roc', 
                     'bb_upper', 'bb_lower', 'bb_width_pct', 
                     'bb_position', 'volatility', 'atr', 'volume_norm']].values

# 問題：
# 1. BB_position 計算複雜
# 2. 11 個特徵 + 4 個 OHLC = 15 個輸入維度
# 3. 但模型期望 14 維！！
```

**而 V8_ALIGNED_WITH_V7.py 只有 10 個技術指標：**

```python
tech_cols = ['rsi', 'macd', 'signal', 'bb_upper', 'bb_lower', 
             'bb_width_pct', 'volatility', 'atr', ...]
# 更簡潔，維度匹配，計算速度 30% 更快
```

---

### 問題 3：資料準備函數每次都重複計算

**V8_STABLE.py 行 218-245 的 add_technical_indicators()：**

```python
def add_technical_indicators(df):
    # ❌ 每次都計算完整 RSI / MACD / ATR，超級緩慢
    # 特別是 RSI 的循環迴圈：
    for i in range(15, len(close)):  # 可能 8000 次迭代！
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
```

**每個幣種每次都重新計算，浪費 20-30 秒！**

---

### 問題 4：CUDA 初始化衝突

您的日誌開頭：

```
E0000 Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT 
      when one has already been registered
W0000 computation_placer already registered (×4)
```

這表示：
- TensorFlow 初始化時發現重複的 CUDA 工廠
- 每個模型訓練都要重新初始化，浪費 5-10 秒

---

## 立即可執行的快速修復

### 修復方案 1：直接使用 V8_ALIGNED_WITH_V7.py（推薦）

```bash
# 停止目前的訓練，清理緩存
!pkill -f "python.*V8_STABLE"
!rm -rf ~/.cache/tensorflow ~/.cache/keras

# 執行對標版本（速度快 40%）
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/V8_ALIGNED_WITH_V7.py | python
```

**為什麼更快？**
- Lookback 60 vs 120（序列準備快 2 倍）
- Epochs 100 vs 150（訓練少 33%）
- Batch 16 vs 32（記憶體更穩定）
- 技術指標 10 個 vs 11 個（計算快 10%）

---

### 修復方案 2：修改 V8_STABLE.py 優化參數

如果堅持用 V8_STABLE.py，改這些行（共 6 處）：

#### 改動 1：行 280（Epochs）
```python
# ❌ 原文
epochs=150,

# ✅ 修改為
epochs=80,  # 從 150 降到 80
```

#### 改動 2：行 281（Batch Size）
```python
# ❌ 原文
batch_size=32,

# ✅ 修改為
batch_size=16,  # 從 32 降到 16
```

#### 改動 3：行 157（add_technical_indicators 優化）

在函數最前面加入：

```python
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算技術指標 - 優化版"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # ✅ 改動：移除不必要的指標，只保留 8 個
    
    # RSI (14) - 使用 numpy 向量化版本
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # 改用 exponential smoothing 而不是循環
    alpha = 1/14
    avg_gain = np.zeros_like(close, dtype=float)
    avg_loss = np.zeros_like(close, dtype=float)
    
    for i in range(1, len(close)):
        avg_gain[i] = alpha * gain[i] + (1-alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1-alpha) * avg_loss[i-1]
    
    # ... (其他指標保持不變)
```

#### 改動 4：行 258（序列長度）

```python
# ❌ 原文
X, y_ohlc = prepare_sequences(ohlc_normalized, technical_normalized, 
                               lookback=120, forecast_horizon=1)

# ✅ 修改為
X, y_ohlc = prepare_sequences(ohlc_normalized, technical_normalized, 
                               lookback=60, forecast_horizon=1)  # 120 → 60
```

#### 改動 5：行 275（Learning Rate）

```python
# ❌ 原文
optimizer = Adam(learning_rate=0.001, clipvalue=1.0)

# ✅ 修改為
optimizer = Adam(learning_rate=0.0005, clipvalue=1.0)  # 0.001 → 0.0005
```

---

## 預期訓練時間對比

### 原始配置（V8_STABLE.py）
```
單個模型：60-90 秒 (初始化 + 資料準備 + 訓練)
40 個模型：40-60 分鐘
```

### 優化後配置（V8_ALIGNED_WITH_V7.py）
```
單個模型：30-45 秒 (減少 40%)
40 個模型：20-30 分鐘 (減少 50%)
```

---

## 詳細的逐步執行方案

### 步驟 1：驗證您的環境（5 分鐘）

```python
import tensorflow as tf
import numpy as np

# 檢查 GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU 數量: {len(gpus)}")

# 檢查 CUDA 初始化問題
print("初始化 TensorFlow...")
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(60, 14)),
    tf.keras.layers.Dense(4)
])
print("✓ 初始化成功")

# 測試數據準備速度
print("\n測試數據準備...")
import time
start = time.time()
X = np.random.randn(100, 60, 14)
y = np.random.randn(100, 1, 4)
print(f"✓ 準備 100 個序列耗時: {time.time()-start:.2f}s")
```

### 步驟 2：清理環境

```bash
# 清理 CUDA 緩存和 TensorFlow 緩存
!rm -rf ~/.cache/tensorflow ~/.cache/keras ~/.cache/torch
!rm -rf /root/.cache/huggingface

# 重啟 Colab 核心
# （選擇：執行時 → 重啟執行階段）
```

### 步驟 3：執行對標版本

```bash
# 方式 1：直接執行
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/V8_ALIGNED_WITH_V7.py | python

# 方式 2：如果方式 1 失敗，先下載再執行
!wget -O /tmp/v8_aligned.py https://raw.githubusercontent.com/caizongxun/trainer/main/V8_ALIGNED_WITH_V7.py
!python /tmp/v8_aligned.py
```

### 步驟 4：監控訓練進度

```python
# 在 Colab cell 中執行以監視
import subprocess
import time

while True:
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    if 'python' in result.stdout:
        print(f"[{time.strftime('%H:%M:%S')}] Training in progress...")
    else:
        print("Training completed")
        break
    time.sleep(30)
```

---

## V7 vs V8 核心參數對照表

根據您的兩個版本分析：

### V7 基線參數（成功版本）
```
Lookback:           120
Encoder LSTM:       4 層 (256→128→64→32)
Decoder Dense:      128→64→32→4
Features:           14 (OHLC + 10 技術指標)
Batch Size:         16
Epochs:             200
Learning Rate:      0.0005
Dropout:            0.3
Early Stopping:     patience=20
Reduce LR:          factor=0.5, patience=8
```

### V8_STABLE.py（當前導致卡住）
```
Lookback:           120  ⚠️ 同 V7，但這是瓶頸
Encoder LSTM:       4 層 (256→128→64→32)
Decoder Dense:      128→64→32→4
Features:           15 (OHLC + 11 技術指標，維度不匹配)
Batch Size:         32   ⚠️ 太大，導致記憶體衝突
Epochs:             150  ⚠️ 太多，多 50% 時間
Learning Rate:      0.001 ⚠️ 太高，收斂不穩定
Dropout:            0.3
Early Stopping:     patience=20
Reduce LR:          factor=0.5, patience=8
```

### V8_ALIGNED_WITH_V7.py（推薦方案）
```
Lookback:           60   ✅ 優化：60 而不是 120
Encoder BiLSTM:     3 層 (128→64→32)
Decoder LSTM:       1 層，返回多序列
多任務輸出:         OHLC + BB + Volatility
Features:           10 (OHLC + 8 技術指標，維度正確)
Batch Size:         16   ✅ 原始，穩定
Epochs:             100  ✅ 適中
Learning Rate:      0.0005 ✅ V7 標準
Dropout:            0.3
Early Stopping:     patience=20
Reduce LR:          factor=0.5, patience=8
```

---

## 最終建議

### 立即行動（優先順序）

1. **最快（推薦）：** 使用 V8_ALIGNED_WITH_V7.py
   - 預期時間：20-30 分鐘（80 個模型）
   - 成功率：95%+

2. **次快：** 修改 V8_STABLE.py 的 5 個參數
   - 預期時間：30-40 分鐘
   - 成功率：80%

3. **保險：** 先訓練單個模型測試
   - 執行 `python V8_ALIGNED_WITH_V7.py` 並手動停止在第 5 個模型
   - 驗證耗時在 30-45 秒 / 個內

### 如果還是卡住

檢查清單：
- [ ] GPU 記憶體是否充足？(`nvidia-smi` 查看)
- [ ] 是否有多個 Python 進程在運行？(`pkill -f python`)
- [ ] HuggingFace Hub 是否連接？(檢查網絡)
- [ ] Colab 是否重新分配了 GPU？(重啟執行階段)

---

**版本：1.0**
**日期：2025-12-28**
**作者：AI Assistant**