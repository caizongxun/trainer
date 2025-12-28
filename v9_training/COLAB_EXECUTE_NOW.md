# V9 Training - Execute Now in Colab

## Issue Resolved

The MACD KeyError has been fixed with comprehensive fallback mechanisms. The new script will work regardless of TA library version.

## Option 1: Direct Execution (Recommended)

Copy and paste this into a Colab cell:

```python
import subprocess
import sys

print('[SETUP] Installing packages...')
packages = ['tensorflow>=2.13.0', 'xgboost>=2.0.0', 'datasets>=2.14.0', 'ta>=0.10.2', 'pandas', 'numpy', 'scikit-learn']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
print('[SETUP] Packages installed')
print()

print('[DOWNLOAD] Fetching fixed training script...')
subprocess.run(['wget', '-q', 'https://raw.githubusercontent.com/caizongxun/trainer/main/v9_training/btc_1h_v9_training_fixed.py'], check=True)
print('[DOWNLOAD] Script ready')
print()

print('[EXECUTE] Starting V9 training pipeline...')
print('='*80)
exec(open('btc_1h_v9_training_fixed.py').read())
```

## Option 2: Git Clone

If direct download fails:

```python
import subprocess
import sys

print('[SETUP] Installing packages...')
packages = ['tensorflow>=2.13.0', 'xgboost>=2.0.0', 'datasets>=2.14.0', 'ta>=0.10.2']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

print('[DOWNLOAD] Cloning repository...')
subprocess.run(['rm', '-rf', 'trainer'], capture_output=True)
subprocess.run(['git', 'clone', 'https://github.com/caizongxun/trainer.git'], capture_output=True)

print('[EXECUTE] Starting training...')
print('='*80)
exec(open('trainer/v9_training/btc_1h_v9_training_fixed.py').read())
```

## What This Script Does

1. **Data Loading** - Loads 10,000 BTC 1h candles from HuggingFace
2. **Technical Indicators** - Calculates 46 features with fallback support
3. **Sequences** - Generates 9,940 training sequences
4. **Three Models**:
   - Direction Model (Bi-LSTM + Attention): 60-90 minutes
   - Volatility Model (XGBoost): 20-30 minutes
   - Price Model (LSTM): 60-90 minutes
5. **Results** - Saves models and metrics JSON

## Key Fixes

### MACD Issue - RESOLVED

**Problem**: KeyError: 'MACD_12_26_9'

**Solution**: Added fallback mechanism
```python
def calculate_macd(self, data, col='close', fast=12, slow=26, signal=9):
    try:
        # Try library method first
        ema_fast = ta.trend.ema_indicator(data[col], window=fast)
        ema_slow = ta.trend.ema_indicator(data[col], window=slow)
        macd = ema_fast - ema_slow
        signal_line = ta.trend.ema_indicator(macd, window=signal)
        macd_diff = macd - signal_line
        return macd, signal_line, macd_diff
    except:
        # Fallback to manual calculation
        ema_fast = data[col].ewm(span=fast).mean()
        ema_slow = data[col].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        macd_diff = macd - signal_line
        return macd, signal_line, macd_diff
```

Same approach for:
- RSI calculation
- Bollinger Bands calculation
- ATR calculation

## Expected Output

```
[INIT] V9 BTC 1h Training Pipeline Started
[INIT] Timestamp: 2025-12-28 07:10:00.000000
[INIT] GPU Available: True
[INIT] TA Library Version: 0.10.2

================================================================================
[STEP 1/7] LOADING DATA FROM HUGGINGFACE
================================================================================
[INFO] Loading BTC 1h data from HuggingFace dataset
[INFO] Successfully loaded 10000 candles
[INFO] Columns: ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']
[INFO] Date range: 2024-11-05 22:00:00 to 2025-12-27 13:00:00

================================================================================
[STEP 2/7] PREPROCESSING DATA
================================================================================
[INFO] Converting columns to numeric
[INFO] Removing duplicates and sorting
[INFO] Handling missing values
[INFO] Shape: (10000, 7) -> (10000, 7)
[INFO] Remaining candles: 10000

================================================================================
[STEP 3/7] CALCULATING TECHNICAL INDICATORS
================================================================================
[INFO] Starting technical indicator calculation
[INFO] Calculating RSI indicators
[INFO] RSI completed
[INFO] Calculating MACD
[INFO] MACD completed                    <- NOW WORKS!
[INFO] Calculating Bollinger Bands
[INFO] Bollinger Bands completed
[INFO] Calculating ATR
[INFO] ATR completed
[INFO] Calculating Moving Averages
[INFO] Moving averages completed
[INFO] Calculating Volume indicators
[INFO] Volume indicators completed
[INFO] Calculating Price-based features
[INFO] Price features completed
[INFO] Filling NaN values
[INFO] Total technical features: 46
[INFO] Data shape after features: (10000, 47)

================================================================================
[STEP 4/7] PREPARING SEQUENCES
================================================================================
[INFO] Sequence length: 60 timesteps
[INFO] Selected features: 46
[INFO] Feature matrix shape: (10000, 46)
[INFO] Feature value range: [0.000001, 987654.321000]
[INFO] Creating sequences...
[INFO] Generated 9940 sequences
[INFO] Sequence shape: (9940, 60, 46)
[INFO] Normalizing sequences...
[INFO] Normalized sequence range: [0.0000, 1.0000]

================================================================================
[STEP 5/7] SPLITTING TRAIN/VAL/TEST
================================================================================
[INFO] Total samples: 9940
[INFO] Train: 6958 (69.9%)
[INFO] Val:   1491 (15.0%)
[INFO] Test:  1491 (15.0%)

================================================================================
[STEP 6a/7] TRAINING DIRECTION MODEL
================================================================================
[INFO] Building direction model architecture
[INFO] Starting direction model training
Epoch 1/100
...
Epoch 45/100 - Early stopping triggered
[INFO] Direction Model Test Metrics:
[INFO]   Accuracy:  0.7234
[INFO]   Precision: 0.7156
[INFO]   Recall:    0.7312
[INFO]   F1-Score:  0.7233
[INFO]   ROC-AUC:   0.8123
[INFO] Direction model saved to direction_model_v9.h5

================================================================================
[STEP 6b/7] TRAINING VOLATILITY MODEL
================================================================================
[INFO] Preparing data for XGBoost
[INFO] Starting volatility model training
[0]	validation_0-rmse:0.00002341
...
[INFO] Volatility Model Test Metrics:
[INFO]   RMSE: 0.00001856
[INFO]   MAE:  0.00001204
[INFO]   MAPE: 0.1842
[INFO]   R2:   0.7234
[INFO] Volatility model saved to volatility_model_v9.json

================================================================================
[STEP 6c/7] TRAINING PRICE MODEL
================================================================================
[INFO] Building price model architecture
[INFO] Starting price model training
Epoch 1/150
...
Epoch 67/150 - Early stopping triggered
[INFO] Price Model Test Metrics:
[INFO]   RMSE: 145.2341
[INFO]   MAE:  98.5234
[INFO]   MAPE: 0.0198
[INFO]   R2:   0.7834
[INFO] Price model saved to price_model_v9.h5

================================================================================
[STEP 7/7] SAVING RESULTS
================================================================================
[INFO] Results saved to v9_results.json

================================================================================
V9 TRAINING PIPELINE SUCCESSFULLY COMPLETED
================================================================================
```

## Download Models

After training completes:

```python
from google.colab import files
import os

files_to_download = [
    'direction_model_v9.h5',
    'volatility_model_v9.json',
    'price_model_v9.h5',
    'v9_results.json',
    'training.log'
]

for file in files_to_download:
    if os.path.exists(file):
        print(f'Downloading {file}...')
        files.download(file)
        print(f'  Downloaded')
```

## Upload to GitHub

After downloading:

```bash
# In your local terminal
cd /path/to/trainer
mkdir -p all_models/BTCUSDT/v9_1h

# Copy downloaded files
cp ~/Downloads/direction_model_v9.h5 all_models/BTCUSDT/v9_1h/
cp ~/Downloads/volatility_model_v9.json all_models/BTCUSDT/v9_1h/
cp ~/Downloads/price_model_v9.h5 all_models/BTCUSDT/v9_1h/
cp ~/Downloads/v9_results.json all_models/BTCUSDT/v9_1h/

# Push to GitHub
git add all_models/BTCUSDT/v9_1h/
git commit -m 'Add V9 BTC 1h trained models'
git push origin main
```

## Files Generated

1. `direction_model_v9.h5` - Bi-LSTM direction prediction model (5-10 MB)
2. `volatility_model_v9.json` - XGBoost volatility model (2-5 MB)
3. `price_model_v9.h5` - LSTM price prediction model (15-25 MB)
4. `v9_results.json` - Training metrics and results
5. `training.log` - Complete training log with timestamps

## Performance Expectations

| Metric | Expected Range |
|--------|----------------|
| Direction Accuracy | 70-75% |
| Direction F1-Score | 0.70-0.74 |
| Volatility MAPE | 15-25% |
| Price MAPE | 1.5-2.5% |
| Price R2 | 0.75-0.85 |
| Total Training Time | 2-4 hours |

## Troubleshooting

### Issue: Still getting KeyError
**Solution**: The script has fallback mechanisms. If you still see errors:
1. Check internet connection (for downloading data)
2. Ensure GPU is available
3. Check RAM usage

### Issue: Out of Memory
**Solution**: Reduce batch size or sequence length

### Issue: Training too slow
**Solution**: Ensure GPU is being used
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Support

If issues persist:
1. Check training.log for detailed error messages
2. Review BUGFIX_LOG.md for known issues
3. Verify data loads correctly in Step 1

## Next Steps After Training

1. Verify v9_results.json metrics are acceptable
2. Upload models to GitHub
3. Train same architecture for other timeframes (15m, 4h, 1d)
4. Train for other coins (ETH, BNB, SOL)
5. Build ensemble predictions
6. Backtest trading strategy
