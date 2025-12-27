# Kaggle 輸出位置設定

## 📂 Kaggle 輸出位置結構

```
/kaggle/working/                    ← 你的 Notebook 的工作目錄 (自動保存)
  ├─ all_models_v7_fast/      ← 訓練的檔案
  │  ├─ BTCUSDT/
  │  │  ├─ BTCUSDT_15m_v7.keras
  │  │  └─ BTCUSDT_15m_v7_params.json
  │  ├─ ETHUSDT/
  │  └─ ...
  └─ training_summary_v7_fast.json

/kaggle/input/                      ← 你上傳的檔案 (單向）

/kaggle/output/ 的介方 → /kaggle/working/ (自動保存)
```

---

## 📁 預設輸出位置

**V7_CLASSIC_FAST.py** 登錙地輸出到當前目錄 `./`：

```python
# 你不需要修改任何东見，Kaggle 會自動保存

# 模型被保存到：
./all_models_v7_fast/

# 訓練統計被保存到：
./training_summary_v7_fast.json
```

---

## ✅ Kaggle 自動保存治理

Kaggle Notebook 的一大羉點：

```
你的 Notebook 工作目錄 (/kaggle/working/)
                ⬇️
        自動保存到 Kaggle 云端
                ⬇️
      你會看到在 Notebook 的 Output 按鈕
```

所以：
- ☯️ 你不需要設定任何东見
- ☯️ 訓練完成了自動就保存好了
- ☯️ 直接從 Kaggle UI 下載就可以

---

## 💡 地位确誊

從 Kaggle Notebook 中執行：

```python
import os

print(f"當前工作目錄: {os.getcwd()}")
print(f"\n訓練檔案位置:")
for folder in os.listdir():
    if folder.startswith('all_models'):
        print(f"  ✓ {folder}/")
        model_count = sum(1 for root, dirs, files in os.walk(folder) for f in files if f.endswith('.keras'))
        print(f"    └ {model_count} 個 .keras 檔案")

if os.path.exists('training_summary_v7_fast.json'):
    print(f"\n  ✓ training_summary_v7_fast.json")
```

---

## 📥 下載訓練結果

### 方法 A: 從 Notebook UI 下載 (推薦新手)

1. 先確保訓練完成 (右上角重控沒有 了)
2. 點擊右上角 **"Output"** 按鈕
3. 選擇 `all_models_v7_fast` 檔案夾
4. 點擊下載 (下位粗粗)
5. 沉拐中下載了 `all_models_v7_fast.zip`

### 方法 B: 從 Notebook 上傳回 HF (最佳)

直接依 Notebook 中執行：

```python
from huggingface_hub import HfApi
import os

# 設定 HF token (可不填也可以)
api = HfApi()

# 上傳整個資料夾
api.upload_folder(
    folder_path='./all_models_v7_fast',
    repo_id='你的-hf-用戶名/trainer-models-v7',  # 改換你的
    repo_type='model',
    private=False  # 設成 True 就是私有
)

print("✓ 檔案已上傳到 HF")
```

### 方法 C: 壓縮檔案 (最小)

```python
import shutil
import os

print("(正在壓縮...")
shutil.make_archive(
    'all_models_v7_fast',
    'zip',
    './',
    'all_models_v7_fast'
)
print("✓ 已壓縮成 all_models_v7_fast.zip")

# 查看檔案大小
file_size_mb = os.path.getsize('all_models_v7_fast.zip') / (1024 * 1024)
print(f"✓ 檔案大小: {file_size_mb:.1f} MB")
```

---

## 📈 訓練統計檔案

**training_summary_v7_fast.json** 有：

```json
{
  "timestamp": "2025-12-28T10:00:00",
  "version": "v7_classic_fast",
  "trained_models": 40,
  "results": [
    {
      "symbol": "BTCUSDT",
      "timeframe": "15m",
      "val_loss": 0.0234,
      "val_mape": 0.45,
      "training_time": 375
    },
    ...
  ]
}
```

---

## 🎉 完整流程

```python
# Cell 1: 安裝依賴
!pip install -q tensorflow keras huggingface-hub pandas scikit-learn psutil

# Cell 2: 克隆並執行
!git clone https://github.com/caizongxun/trainer.git
%cd trainer
!python V7_CLASSIC_FAST.py

# Cell 3: 检查輸出檔案
import os
print("\n訓練結果：")
print(f"  ✓ 檔案數量: {len(os.listdir('all_models_v7_fast'))} 個")
print(f"\n檔案位置：")
print(f"  /kaggle/working/all_models_v7_fast/")
print(f"  /kaggle/working/training_summary_v7_fast.json")

# Cell 4: 上傳回 HF (可選)
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='./all_models_v7_fast',
    repo_id='你的-hf-repo/trainer-models-v7',
    repo_type='model'
)
print("\n✓ 檔案已上傳到 HF")

# Cell 5: 下載結果 (伊物东)
print("\n✓ Notebook 完成！")
print("\n下一步：")
print("  1. 點擊 Notebook 上方 'Output' 按鈕")
print("  2. 選擇 'all_models_v7_fast' 資料夾")
print("  3. 點擊下載圖示")
```

---

## 🏗️ 故 Troubleshooting

### Q: 輸出檔案在哪？

```python
# Notebook 中執行
import os
for root, dirs, files in os.walk('.'):
    if 'all_models' in root:
        print(f"{root}: {len(files)} 檔案")
```

### Q: 檔案沒有自動保存？

Kaggle 自動保存 `/kaggle/working/`中的所有檔案。

但如果 Notebook 幸運（护会實繖！），你也可以：

```python
# 主動上傳到 HF
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='./all_models_v7_fast',
    repo_id='你的-repo',
    repo_type='model'
)
```

### Q: 檔案太大，下載不了？

```python
# 壓縮并上傳到 HF
import shutil
shutil.make_archive('models', 'zip', '.', 'all_models_v7_fast')

from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='models.zip',
    path_in_repo='all_models_v7_fast.zip',
    repo_id='你的-repo',
    repo_type='model'
)
```

---

## 📝 穂位

| 位置 | 読寫 | 寫入 | 自動保存 |
|---------|--------|--------|----------|
| `/kaggle/working/` | ✅ | ✅ | ✅ (云端) |
| `/kaggle/output/` | ✅ | ✅ | ✅ (下載區) |
| `/kaggle/input/` | ✅ | ❌ | ✅ (徊輸區) |

---

## ✅ 最粀漚的做法

1. 訓練程床之最二重樹尾一個 Cell
2. 複製這個：

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='./all_models_v7_fast',
    repo_id='your-username/trainer-models-v7',
    repo_type='model'
)
print("Done!")
```

3. 扨上粗粗執行
4. 後然剪那個 Cell 執行
5. 檔案自動上傳到 HF 了，永久保存

---

祝你訓練順利！ 🚀
