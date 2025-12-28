# QUICK_START_V9.md

本文件提供 V9 多步預測（30 根 K 棒 -> 預測後 10 根）模型的最短操作路徑。

## 目標

- 以前 30 根 K 棒（含多種價格特徵）預測未來 10 根 K 棒的價格軌跡（multi-horizon / MIMO）。
- 額外加入波動率（range/ATR proxy）輔助頭，讓模型學到波動幅度。
- 以 GPU 訓練，最多 100 epochs，但使用 EarlyStopping + TimeBudget 控制總時間在 2 小時內。
- 輸出模型放在 Colab 本地 `./all_models/models_v9/`，最後以「整個資料夾」一次上傳到 HuggingFace，避免逐檔上傳觸發 API 限制。

## 一行命令（Colab）

在 Colab 新筆記本中，確保 Runtime 使用 GPU（Runtime -> Change runtime type -> GPU），然後執行：

```bash
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v9.py | python
```

## 重要輸出

- 訓練後模型：`./all_models/models_v9/{SYMBOL}/{SYMBOL}_{INTERVAL}_v9.keras`
- 每個模型的 scaler 與欄位：`..._v9_scaler.json`
- 每個模型的訓練摘要（含 val_mape_close）：`..._v9_meta.json`
- 全部彙總：`./all_models/training_summary_v9.json`

## 可選參數

```bash
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v9.py | python -- \
  --seq_len 30 \
  --pred_len 10 \
  --epochs 100 \
  --time_budget_min 120 \
  --intervals 15m,1h \
  --max_models 0
```

- `--max_models 0` 代表不限制（全跑）。

## 上傳到 HuggingFace

腳本第 6 步會要求輸入 HuggingFace token。

- 直接 Enter：跳過上傳。
- 輸入 token：會使用 `upload_folder()` 將本地 `./all_models/models_v9/` 一次上傳到 dataset repo `zongowo111/cpb-models` 的 `models_v9/` 目錄。

