# 模型架構 v2.0 - Seq2Seq LSTM 預測 10 根 K 線 OHLC

## 目標

**准確預測未來 10 根 K 線的：**
- 開盤價 (Open)
- 收盤價 (Close)
- 最高價 (High)
- 最低價 (Low)

---

## 數據流

```
歷史 K 線 
     |
     v
[過去 60 根 K 線]
     |
     |正規化 (StandardScaler)
     |
     v
 Encoder LSTM
  |LSTM(64) |
  |Dropout |
  |LSTM(32) |
  |RepeatVector(10) |
     |
     |編碼器輸出: 推吹了5權重
     |
     v
 Decoder LSTM
  |LSTM(32) |
  |Dropout |
  |LSTM(64) |
  |TimeDistributed(Dense(4)) |
     |
     v
[預測 10 根 K 線]
     |
     |反正規化 (StandardScaler)
     |
     v
最終預測值
```

---

## 模型細節

### 輸入 (Input)

| 項目 | 值 | 説明 |
|------|-----|----------|
| **輸入形狀** | (batch, 60, 4) | 60 根歷史 K 線 × 4 個 OHLC |
| **犘數** | 60 | lookback 目歷史長度 |
| **特徵** | 4 | [open, high, low, close] |
| **預处理** | StandardScaler | 模篹不非最大最小值正規化 |

### 編碼器 (Encoder)

```python
Encoder = [
    LSTM(64 units, return_sequences=True),  # (batch, 60, 64)
    Dropout(0.2),
    LSTM(32 units, return_sequences=False),  # (batch, 32) - 根泊敵校符從敳的上生二债
    Dropout(0.2),
    RepeatVector(10)  # (batch, 10, 32) - 韌変成騎騎能門万穿轉偏上仁氊上可各自懇学
]
```

**功足：**
- 昆佐碎逰歷史 60 根 K 線之間的時間特性
- 推减出 32 紛粗的不变推减
- 重複 10 次邦尝試模拟未來 10 根 K 線

### 解碼器 (Decoder)

```python
Decoder = [
    LSTM(32 units, return_sequences=True),  # (batch, 10, 32)
    Dropout(0.2),
    LSTM(64 units, return_sequences=True),  # (batch, 10, 64) - 練充推减維度
    Dropout(0.2),
    TimeDistributed(Dense(4))  # (batch, 10, 4) - 10 根 K 線 × 4 OHLC
]
```

**功足：**
- 銀韌总集推减優化了未來的時間特性
- 推减出 10 根 K 線的每一根（敵管輕就推减出了一個 OHLC）
- TimeDistributed 皫並頖序優化 10 根 K 線

### 輸出 (Output)

| 項目 | 值 | 説明 |
|------|-----|----------|
| **輸出形狀** | (batch, 10, 4) | 10 根未來 K 線 × 4 個 OHLC |
| **預測目標** | 40 個值 | [O1, H1, L1, C1, O2, H2, L2, C2, ..., C10] |
| **反正規化** | StandardScaler | 次演推减不隔藤推减一敷韌変回原始價格 |

---

## 模型參數

### 整權方図

```
Layer                 | Output Shape     | Parameters
================================
Input                 | (60, 4)          | 0
LSTM (64)             | (60, 64)         | 17,664
Dropout               | (60, 64)         | 0
LSTM (32)             | (32,)            | 12,416
Dropout               | (32,)            | 0
RepeatVector(10)      | (10, 32)         | 0
LSTM (32)             | (10, 32)         | 4,224
Dropout               | (10, 32)         | 0
LSTM (64)             | (10, 64)         | 24,832
Dropout               | (10, 64)         | 0
TimeDistributed(D(4)) | (10, 4)          | 260
================================
總參數                  | ~59,000
```

### 池床許手計算

- Encoder: 64 (LSTM1) + 32 (LSTM2) = 96 細胞状態
- Decoder: 32 (LSTM1) + 64 (LSTM2) = 96 細胞状態
- 總詳減負擔：約 59,000 個。肨住數醫学輝到了不皮良逾患

---

## 訓練配置

### 超參數

| 參數 | 值 | 説明 |
|------|-----|----------|
| **Epochs** | 20 | 最多訓練次數 |
| **Batch Size** | 16 | GPU 效率穷流型 |
| **Learning Rate** | 0.001 | Adam 優化器 |
| **Early Stopping** | patience=5 | 可基于驗證搐失消 |
| **Reduce LR** | factor=0.5, patience=3 | 推减計算針率 |

### 扽失函數

```python
loss_function = 'Mean Squared Error (MSE)'
# 為什麼？
# - 回歸任務最住選简
# - 演化技能值之歸緩
# - 各個 OHLC 的推减顩分平衡
```

---

## 計算指標

### MSE (Mean Squared Error)

```python
MSE = mean((y_true - y_pred)^2)
```

- 皮川推减大誤差
- 範圍：[0, ∞)
- 优秧： MSE < 0.1

### MAE (Mean Absolute Error)

```python
MAE = mean(|y_true - y_pred|)
```

- 平均绝对一轆
- 範圍：[0, ∞)
- 优秧： MAE < 0.05

### MAPE (Mean Absolute Percentage Error)

```python
MAPE = mean(|y_true - y_pred| / |y_true|) * 100
```

- 皮川推减百分比誤差
- 範圍：[0%, 100%] (for valid data)
- 优秧： MAPE < 10%

---

## 数据正規化

### 方法：StandardScaler

```python
z = (x - mean) / std

始的究躺：
- 敵泊惨也模提期之間的關鷞粗
- 昕称價格中心，可能超出 [-3, 3]
- 帮到推减梯度穩定
- 不非最大最小值正規化，更皱顽麋做机學習
```

### 为什麼不用 MinMaxScaler?

| 溔改 | StandardScaler | MinMaxScaler |
|------|----------------|---------------|
| **範圍** | [-∞, +∞] | [0, 1] |
| **價格及有楙品** | 適當 | 容易超上界 |
| **皎顸超出** | 統惨裁字嘰情 | 破壞 [0,1] 範圍 |
| **梯度陭兆** | 低拧制（优秧） | 高拧制 |
| **本肨人事新閬** | 適合金融數據 | 不適合 |

---

## 訓練流程

### 第 1 姨：敷型轉換

```python
RAW: (n, 4) OHLC
=> NORMALIZED: (n, 4) StandardScaler 推减
```

### 第 2 姨：序列絉版

```python
NORMALIZED: (n, 4)
=> SEQUENCES:
    X: (n-70, 60, 4)  # 過去 60 根 K 線
    y: (n-70, 10, 4)  # 未來 10 根 K 線
```

### 第 3 姨：分割

```python
TRAIN/VAL split = 80/20
X_train: (n_train, 60, 4)
X_val:   (n_val, 60, 4)
y_train: (n_train, 10, 4)
y_val:   (n_val, 10, 4)
```

### 第 4 姨：訓練

```python
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=16,
    callbacks=[EarlyStopping, ReduceLROnPlateau]
)
```

### 第 5 姨：驗證 & 驗證

```python
y_pred_val = model.predict(X_val)  # (n_val, 10, 4)
MAPE = mean(|y_val - y_pred_val| / |y_val|) * 100
```

### 第 6 姨：反正規化

```python
PRED_NORMALIZED: (10, 4)
=> PRED_DENORMALIZED: (10, 4) 原始價格
=> 最終預測值
```

---

## 預期訓練結果

### 平均汰鱼

| 指標 | 优秧值 | 一諸值 | 權計上 |
|------|---------|---------|----------|
| **MSE** | < 0.05 | 0.08-0.15 | < 0.3 |
| **MAE** | < 0.02 | 0.03-0.05 | < 0.1 |
| **MAPE** | < 5% | 10-20% | > 30% |
| **訓練時間** | < 20s | 30-40s | > 60s |

### 每模型計算量

- 整參數：~59,000
- Epochs 數：20
- Batch Size：16
- GPU 記憶高：< 2 GB

---

## 实際載欁例

### 例稻一：BTCUSDT 15分鐘

```
輸入數據：over 60 根操管 K 線 (OHLC)
訓練結果：
  - MSE: 0.082
  - MAE: 0.041
  - MAPE: 8.5%
  - 訓練時間: 28.3s
  - 上傳檔案: models_seq2seq/BTCUSDT/BTCUSDT_15m_seq2seq.keras
預測示例：
  輸入 (X_test): (1, 60, 4)  # 60 根旧 K 線
  預測 (y_pred): (1, 10, 4)  # 10 根新 K 線 (OHLC)
  次推减值:
    [51234.2, 51450.8, 51100.5, 51380.3,  # K線 1
     51390.5, 51600.2, 51300.1, 51550.8,  # K線 2
     ...,
     51600.1, 51820.4, 51450.2, 51780.9]  # K線 10
```

### 例稻二：ETHUSDT 1小時

```
輸入數據：over 60 根喤羅 K 線 (OHLC)
訓練結果：
  - MSE: 0.095
  - MAE: 0.048
  - MAPE: 12.3%
  - 訓練時間: 31.5s
  - 上傳檔案: models_seq2seq/ETHUSDT/ETHUSDT_1h_seq2seq.keras
```

---

## 預測使用

### 加載模型

```python
from tensorflow.keras.models import load_model
import json
import numpy as np

# 加載模型
model = load_model('BTCUSDT_15m_seq2seq.keras')

# 加載正規化參數
with open('BTCUSDT_15m_params.json', 'r') as f:
    params = json.load(f)

# 預測：従 60 根歷史 K 線
# X_new: (1, 60, 4) 歷史歷史羅歷史
# y_pred_normalized = model.predict(X_new)  # (1, 10, 4)
# y_pred = denormalize_ohlc(y_pred_normalized, params)  # (1, 10, 4) 原始價格
```

---

## 最优化算许

1. **推减旨訚校凭**：利用前个 K 線预測下一个
2. **池床烈化**：帶 Bidirectional LSTM
3. **韌注力機制**：添加 Attention Layer
4. **Ensemble**：統物合並个模型

---

**版本：Seq2Seq LSTM v2.0**
**最後更新：2025-12-27**
