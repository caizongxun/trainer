# Kaggle 訓練設定指南

## 🎯 Kaggle 優勢

- ✅ **免費 GPU** (Tesla T4 或 P100)
- ✅ **沒有時間限制** (週期 12 小時，可續)
- ✅ **預裝了 TensorFlow + CUDA**
- ✅ **自動保存輸出**
- ❌ 比 Colab 慢，但穩定

---

## 📋 步驟 1：建立 Kaggle 帳號

1. 去 https://www.kaggle.com
2. 註冊帳號 (用 Google/GitHub 最快)
3. 驗證郵件

---

## 🔑 步驟 2：建立 API Token

1. 登入 Kaggle
2. 點擊右上角頭像 → "Account" → "Settings"
3. 向下滾動到 "API" 部分
4. 點擊 "Create New API Token"
5. 會下載 `kaggle.json`

---

## 🚀 步驟 3：建立 Kaggle Notebook

### 方法 A：用網頁介面 (推薦新手)

1. 去 https://www.kaggle.com/notebooks
2. 點擊 "+ New Notebook"
3. 選擇 "Python" 環境
4. **啟用 GPU**：點擊右上角 "⚙️ Settings" → "Accelerator" → "GPU T4 x2"
5. 保存 Notebook

### 方法 B：用 CLI (推薦進階)

```bash
# 1. 安裝 kaggle CLI
pip install kaggle

# 2. 上傳 API Token
# 在 ~/.kaggle/ 目錄放 kaggle.json
# Windows: C:\Users\<username>\.kaggle\kaggle.json
# Mac/Linux: ~/.kaggle/kaggle.json

# 3. 建立 Notebook
kaggle notebooks create -f V7_CLASSIC_FAST.py -j trainer-v7 -c crypto-price-prediction
```

---

## 📝 步驟 4：在 Kaggle Notebook 執行

### 簡單版本（推薦）

在 Kaggle Notebook 中執行：

```python
# Cell 1: 安裝依賴
!pip install -q tensorflow keras huggingface-hub pandas scikit-learn psutil

# Cell 2: 克隆倉庫
!git clone https://github.com/caizongxun/trainer.git
%cd trainer

# Cell 3: 執行訓練
!python V7_CLASSIC_FAST.py
```

### 完整版本（帶 GPU 檢查）

```python
# Cell 1: GPU 檢查
import tensorflow as tf

print("GPU 設備：")
print(tf.config.list_physical_devices('GPU'))
print(f"\n可用 GPU 數：{len(tf.config.list_physical_devices('GPU'))}")

# Cell 2: 安裝依賴
!pip install -q tensorflow keras huggingface-hub pandas scikit-learn psutil

# Cell 3: 克隆倉庫
!git clone https://github.com/caizongxun/trainer.git
%cd trainer

# Cell 4: 執行訓練（Kaggle 專用版）
!python V7_CLASSIC_FAST.py

# Cell 5: 輸出檔案位置
import os
print("\n訓練完成！檔案位置：")
print(os.listdir('./all_models_v7_fast')[:5])  # 顯示前 5 個
```

---

## ⏱️ Kaggle 訓練時間

| GPU | 時間 |
|-----|------|
| **Tesla T4 (Kaggle 免費)** | **15-20 小時** |
| **Tesla P100 (Kaggle Plus)** | **8-10 小時** |
| RTX 3050 (本地) | 8-12 小時 |

**注意**：Kaggle 每個 Notebook 有 12 小時執行時間限制。

40 個模型需要 15-20 小時，所以需要分次執行：

```python
# 第一次執行：訓練前 20 個
max_pairs = min(20, len(pairs_to_train))

# 第二次執行：訓練後 20 個
max_pairs = min(20, len(pairs_to_train))
start_index = 20
```

---

## 📤 步驟 5：下載訓練結果

### 從 Kaggle Notebook 下載

1. 右上角點 "Output" 按鈕
2. 選擇 `all_models_v7_fast` 資料夾
3. 點擊下載圖示

### 或用代碼下載

```python
import shutil

# 壓縮模型資料夾
shutil.make_archive('models', 'zip', '.', 'all_models_v7_fast')

# Kaggle 會自動把它放在 Output
print("Notebook Output 中會出現 models.zip")
```

---

## 🚨 常見問題

### Q：GPU 沒有被偵測

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# 如果是空，表示沒啟用 GPU
# 回到 Notebook Settings，確保 "GPU T4 x2" 已選中
```

### Q：執行超過 12 小時怎辦

**解決方案：分次執行**

```python
# Notebook 1：訓練前 20 個模型
max_pairs = 20

# 執行完成後
# 建立新 Notebook 2：訓練後 20 個模型
# 從 HF 下載已訓練的模型，繼續訓練
```

### Q：Kaggle 的模型怎麼上傳回 HF

```python
# Kaggle Notebook 中
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path='./all_models_v7_fast',
    repo_id='你的-hf-用戶名/你的-repo',
    repo_type='model'
)
```

---

## 📋 完整 Kaggle Notebook 代碼

複製粘貼這個到 Kaggle Notebook 的第一個 Cell：

```python
# 安裝依賴
import subprocess
import sys

packages = [
    'tensorflow',
    'keras', 
    'huggingface-hub',
    'pandas',
    'scikit-learn',
    'psutil'
]

for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

print("✓ 所有依賴已安裝")

# 檢查 GPU
import tensorflow as tf
print(f"\n✓ 偵測到 {len(tf.config.list_physical_devices('GPU'))} 個 GPU")
print(f"✓ TensorFlow 版本: {tf.__version__}")
```

第二個 Cell：

```bash
# 克隆倉庫
git clone https://github.com/caizongxun/trainer.git
cd trainer

# 執行訓練
python V7_CLASSIC_FAST.py
```

---

## 🎯 快速開始

1. **5 分鐘**：建立 Kaggle 帳號 + API Token
2. **2 分鐘**：建立 Notebook + 啟用 GPU
3. **1 分鐘**：複製粘貼上面的代碼
4. **15-20 小時**：訓練執行中
5. **5 分鐘**：下載結果

**總共：30 分鐘設定 + 15-20 小時訓練**

---

## 💡 Kaggle vs 本地

| 項目 | Kaggle | 本地 RTX 3050 |
|------|--------|---------------|
| **GPU** | T4 (免費) | RTX 3050 |
| **速度** | 慢 (15-20h) | 快 (8-12h) |
| **設定** | 簡單 | 需要 CUDA |
| **成本** | 免費 | 電費 |
| **缺點** | 12h 時間限制 | 需要高端 GPU |

---

## 推薦

**如果你的本地 GPU 沒有 CUDA：**
- 用 **Kaggle**（更簡單）

**如果你已經有 CUDA 環境：**
- 用 **本地 RTX 3050**（更快）

**如果你想要最快：**
- 用 **本地 RTX 4090/3090**（1-2 小時）

---

祝訓練順利！有問題再問！ 🚀
