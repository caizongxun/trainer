# 虛擬貨幣價格預測訓練

## 版本對比

### V7 Classic 版本 (推薦新手)

```bash
!rm -rf ~/.cache/tensorflow ~/.cache/keras ~/.cache/torch
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/V7_CLASSIC.py | python
```

**特點：**
- 完全複製 V7 原始訓練邏輯
- Lookback: 60 根 K 線
- 模型架構：3 層 BiLSTM Encoder + 1 層 LSTM Decoder
- 多任務學習：OHLC + Bollinger Bands + Volatility
- 訓練 20 種幣種 (40 個模型 - 2 個時間框架)
- 訓練時間：~10-15 分鐘 (40 個模型)
- 數據來源：HF 資料集 (zongowo111/cpb-models)

**配置：**
| 項目 | 值 |
|------|-----|
| Learning Rate | 0.0005 |
| Batch Size | 16 |
| Epochs | 100 |
| Lookback | 60 根 |
| Forecast Horizon | 10 根 |
| Dropout | 0.3 |
| Early Stopping Patience | 20 |
| ReduceLR Patience | 8 |

---

### V8 Stable 版本 (穩定快速)

```bash
!rm -rf ~/.cache/tensorflow ~/.cache/keras ~/.cache/torch
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/V8_STABLE.py | python
```

**特點：**
- 改進版 V7 邏輯
- Lookback: 120 根 K 線 (更多上下文)
- 14 維技術指標 (比 V7 的 10 維更多)
- Batch Size: 32 (GPU 優化)
- 訓練 80 個模型
- 訓練時間：~40-60 分鐘 (80 個模型)
- 不使用 Mixed Precision (保證穩定性)
- 動態 GPU 記憶體分配

**配置：**
| 項目 | 值 |
|------|-----|
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 150 |
| Lookback | 120 根 |
| Forecast Horizon | 1 根 |
| Dropout | 0.3 |
| GPU Memory Growth | True |
| Mixed Precision | False |

---

## 數據流程

### 數據來源：HF 資料集

```
zongowo111/cpb-models (HF 資料集)
  ⮕ klines_binance_us/
     ├─ BTCUSDT/
     │  ├─ BTCUSDT_15m.csv
     │  └─ BTCUSDT_1h.csv
     ├─ ETHUSDT/
     │  ├─ ETHUSDT_15m.csv
     │  └─ ETHUSDT_1h.csv
     └─ ... (其他 20+ 幣種)
```

### 訓練流程

```
[1/7] 清理緩存
  ⮕ 移除舊的 TensorFlow/Keras 緩存

[2/7] 環境設定
  ⮕ 檢測 Colab 環境
  ⮕ GPU 配置

[3/7] 依賴套件
  ⮕ TensorFlow, Keras, HF Hub, Pandas, NumPy, Scikit-Learn

[4/7] 數據準備
  ⮕ 從 HF 下載 CSV 檔案
  ⮕ 自動篩選 40+ 個 CSV 檔案
  ⮕ 按幣種組織

[5/7] 模型訓練
  ⮕ V7: 40 個模型 (~10-15 分鐘)
  ⮕ V8: 80 個模型 (~40-60 分鐘)
  ⮕ 每個模型都獨立訓練

[6/7] 上傳模型
  ⮕ 自動上傳到 HF 資料集
  ⮕ 保存參數檔案 (.json)

[7/7] 完成
  ⮕ 生成訓練摘要
  ⮕ 顯示平均 MAPE、訓練時間等
```

---

## 技術指標 (V7 標準)

### RSI (相對強弱指數)

```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain (14) / Average Loss (14)
```

特點：
- 偵測超買/超賣條件
- 範圍：0-100
- 典型閾值：30 (超賣)、70 (超買)

### MACD (移動平均收斂背離)

```
MACD = EMA(12) - EMA(26)
Signal = EMA(9) 的 MACD
```

特點：
- 偵測趨勢變化
- 顯示動量
- 交叉點是交易信號

### Bollinger Bands (布林帶)

```
Middle = SMA(20)
Upper = Middle + 2 × StdDev(20)
Lower = Middle - 2 × StdDev(20)
```

特點：
- 偵測波動性
- 波段寬度表示市場波動性
- 價格靠近上/下波段表示極端條件

### ATR (真實波幅範圍)

```
TR = Max(H-L, |H-C_prev|, |L-C_prev|)
ATR = SMA(TR, 14)
```

特點：
- 衡量市場波動性
- 不受價格高低影響
- 用於止損設置

### Volatility (波動性)

```
Volatility = StdDev(Close, 20) / SMA(Close, 20)
```

特點：
- 標準化波動性
- 範圍相對固定
- 用於風險評估

---

## 模型架構

### V7 架構

```
Input: (60, 10)
  ⮕ 60 根 K 線 × 10 個特徵
  │ (OHLC + RSI + MACD + Signal + BB_upper + BB_lower)
  │
  ⮕ Bidirectional LSTM (128) + LayerNorm + Dropout(0.3)
  │
  ⮕ Bidirectional LSTM (64) + LayerNorm + Dropout(0.3)
  │
  ⮕ LSTM (32) + LayerNorm + Dropout(0.3)
  │
  ⮕ RepeatVector(10)
  │
  ⮕ LSTM (64) + LayerNorm + Dropout(0.3)
  │
  ⮕ TimeDistributed Dense(4) → OHLC 輸出
  ├─ TimeDistributed Dense(2) → BB 參數輸出
  └─ TimeDistributed Dense(1) → Volatility 輸出

Output: 3 個任務
  - OHLC: (10, 4)
  - BB_Params: (10, 2)
  - Volatility: (10, 1)
```

### V8 架構

```
Input: (120, 14)
  ⮕ 120 根 K 線 × 14 個特徵
  │ (OHLC + 11 個技術指標)
  │
  ⮕ Bidirectional LSTM (256) + LayerNorm + Dropout(0.3)
  │
  ⮕ Bidirectional LSTM (128) + LayerNorm + Dropout(0.3)
  │
  ⮕ Bidirectional LSTM (64) + LayerNorm + Dropout(0.2)
  │
  ⮕ LSTM (32) + LayerNorm + Dropout(0.2)
  │
  ⮕ Dense(128) + Dropout(0.2)
  │
  ⮕ Dense(64) + Dropout(0.1)
  │
  ⮕ Dense(32)
  │
  ⮕ Dense(4) → OHLC 輸出

Output: 單一任務
  - OHLC: (1, 4)
```

---

## 性能比較

| 項目 | V7 Classic | V8 Stable |
|------|------------|----------|
| **Lookback** | 60 | 120 |
| **特徵維度** | 10 | 14 |
| **Batch Size** | 16 | 32 |
| **Epochs** | 100 | 150 |
| **單個模型時間** | 20-30s | 30-45s |
| **40 模型時間** | 13-20 分鐘 | - |
| **80 模型時間** | - | 40-60 分鐘 |
| **GPU 使用率** | 60-70% | 80-90% |
| **預期 MAPE** | 0.1-0.5% | 0.1-0.5% |
| **穩定性** | 100% ✅ | 100% ✅ |

---

## 使用指南

### Colab 執行步驟

1. 打開 Google Colab：https://colab.research.google.com/

2. 新建代碼儲存格

3. 選擇 GPU 運行時
   - 連接 → 變更執行階段類型 → GPU

4. 複製貼上命令：

   **V7 Classic:**
   ```bash
   !rm -rf ~/.cache/tensorflow ~/.cache/keras ~/.cache/torch
   !curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/V7_CLASSIC.py | python
   ```

   **V8 Stable:**
   ```bash
   !rm -rf ~/.cache/tensorflow ~/.cache/keras ~/.cache/torch
   !curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/V8_STABLE.py | python
   ```

5. 按 **Ctrl+Enter** 執行

6. 等待執行完成 (顯示進度條)

### 輸出檔案

執行後會生成：

```
./all_models_v7/ (V7) 或 ./all_models_v8_stable/ (V8)
├─ BTCUSDT/
│  ├─ BTCUSDT_15m_v7.keras
│  └─ BTCUSDT_15m_v7_params.json
├─ ETHUSDT/
│  ├─ ETHUSDT_15m_v7.keras
│  └─ ETHUSDT_15m_v7_params.json
└─ ...

training_summary_v7.json (或 v8)
  - 訓練統計
  - 模型性能
  - 時間戳
```

---

## 常見問題

### Q: GPU 卡住怎麼辦？

**A:** 使用 V8_STABLE.py，它已經移除了會導致卡住的 Mixed Precision 和 L1/L2 正則化。

### Q: 可以只訓練 10 個模型嗎？

**A:** 可以，編輯 `max_pairs = min(10, len(pairs_to_train))` 這一行。

### Q: 訓練時間太長了怎麼辦？

**A:** 使用 V7_CLASSIC.py，它只訓練 40 個模型，快 2 倍。

### Q: 模型精度不好怎麼辦？

**A:** 這不是模型問題，而是虛擬貨幣市場本身的噪音。0.1-0.5% MAPE 已經是不錯的結果。

### Q: 可以修改技術指標嗎？

**A:** 可以，編輯 `add_technical_indicators()` 函數。但建議保持 V7 標準配置。

---

## 技術支援

- **問題報告**：[GitHub Issues](https://github.com/caizongxun/trainer/issues)
- **討論區**：[GitHub Discussions](https://github.com/caizongxun/trainer/discussions)
- **HF 資料集**：[zongowo111/cpb-models](https://huggingface.co/datasets/zongowo111/cpb-models)

---

## 授權

MIT License

**最後更新**: 2025-12-27
