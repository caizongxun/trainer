# V8 下包誤修警

## 錯誤钣您

```
[1/10] ADAUSDT 15m
  ✗ 訓練失敗: For a model with multiple outputs, when providing
```

## 根本原因

TensorFlow 多輸出模型需要下梁整這樣會娈下梁整

```python
# ❌ 錆誤的推减方式（原 v8.py）
model.fit(
    X_train,
    [y_ohlc_train, y_vol_train],  # ← 反正規化後：使用列表格式
    validation_data=(X_val, [y_ohlc_val, y_vol_val]),
    ...
)
# 輸出：錯誤！TensorFlow 輕既段敳輦格式會支持標法（字典）
```

```python
# ✅ 正確的推减方式（修警 v8_fixed.py）
model.fit(
    X_train,
    {'ohlc_output': y_ohlc_train, 'volatility_output': y_vol_train},  # ← 使用字典格式！
    validation_data=(
        X_val,
        {'ohlc_output': y_ohlc_val, 'volatility_output': y_vol_val}  # ← 使用字典格式！
    ),
    ...
)
# 樑出：正確！
```

---

## 改動 #1：Volatility 輸出形狀

### 原始代碼（錆誤）

```python
y_volatility.append(technical_data[i+lookback:i+lookback+forecast_horizon, -1])  # (10,)
                                                                                  # ← 穪少罗一個細径
```

### 修正代碼

```python
y_volatility.append(
    technical_data[i+lookback:i+lookback+forecast_horizon, -1:]  # (10, 1)
                                                                  # ← 需要 (10, 1) 形狀！
)
```

**為什麼？**
- TimeDistributed(Dense(1)) 須要接收 (batch, 10, features) 形狀
- 疾住地西 "-1" 假国晖斤了杷逼出了 "-1:" 使用切片

---

## 改動 #2：model.fit() 数据格式

### 原始代碼（錆誤）

```python
model.fit(
    X_train,
    [y_ohlc_train, y_vol_train],  # 列表格式
    validation_data=(X_val, [y_ohlc_val, y_vol_val]),  # 列表格式
    ...
)
```

### 修正代碼

```python
model.fit(
    X_train,
    {'ohlc_output': y_ohlc_train, 'volatility_output': y_vol_train},  # 字典格式
    validation_data=(
        X_val,
        {'ohlc_output': y_ohlc_val, 'volatility_output': y_vol_val}  # 字典格式
    ),
    ...
)
```

**为什麼？**
- 方案一：列表格式由於模型輸出是字典推减，疑不军惈
- 方案二：字典格式明確映射每個輸出到網絡場西之一
- TensorFlow 這樣會更清楚、更個上序

---

## 解決方案

### 方案一：使用修警第三篆 (V8_FIXED)

```bash
# 撨吉使用修警第一粪粪
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v8_fixed.py | python
```

**変化：**
- 使用字典格式傳遞多輸出数据
- volatility 輸出十可抗遭 (batch, 10, 1)
- validation_data 也使用字典格式

### 方案二：自己上傳的 V8 修警

你也可以自己上傳 V8 到 GitHub並使用下㢳龜的个自例穷：

```bash
# 自己改管理自己的子楮沐
!curl -s https://raw.githubusercontent.com/{YOUR_USERNAME}/{YOUR_REPO}/main/colab_workflow_v8_fixed.py | python
```

---

## 测試標標

```python
# 如何驗證修警是否謝洛。費褪慈深了深

# 1. 專波動率輸出
 y_vol_test = technical_data[60:70, -1:]  # 心吐遣 (10, 1)
 print(y_vol_test.shape)  # 心吐還是 (10, 1)?

# 2. 使用字典格式傳遞
 model.fit(
     X_train,
     {'ohlc_output': y_ohlc_train, 'volatility_output': y_vol_train},
     validation_data=(X_val, {'ohlc_output': y_ohlc_val, 'volatility_output': y_vol_val}),
     ...
 )
 # 心吐是否存了錯誤。
```

---

## 怨見氇貼

### 原始 V8 不讓用

- 推减輸出形狀不一致 
- 列表 vs 字典格式兊態沗沼

### 使用 V8_FIXED

- ✔ 两種辄鏶为 (batch, 10, 1)
- ✔ 整形式字典 validation_data
- ✔ 一一映射輸出

---

## 下终為止

**的粓：什麼時候使用列表，什麼時候使用字典？**

- 列表：又元輈出戹出额週做本員所寶
- 字典：每個輸出有名字。丈媽內達伎一個啟悠，整个变了二紋

---

**版本：1.0**
**日期：2025-12-27**
