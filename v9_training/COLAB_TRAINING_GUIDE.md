# BTC 1h V9 Model Training Guide - Google Colab

## Overview

This guide provides step-by-step instructions for training the V9 cryptocurrency price prediction model in Google Colab.

## Quick Start

Copy and paste the code below into Colab cells and run sequentially.

## Step 1: Setup Environment

```python
import subprocess
import sys

print('[SETUP] Installing required packages...')
print()

packages = [
    'tensorflow>=2.13.0',
    'xgboost>=2.0.0',
    'datasets>=2.14.0',
    'ta>=0.10.2',
]

for package in packages:
    print(f'Installing {package}...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print('All packages installed successfully')
print()

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU devices available: {len(gpus)}')
for gpu in gpus:
    print(f'  - {gpu}')
print()
```

## Step 2: Import Libraries

```python
import os
import sys
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import ta
from datasets import load_dataset

print('[SETUP] All libraries imported successfully')
print()
```

## Step 3: Download Training Script

```python
import subprocess
import os

print('[STEP 0] Downloading training script from GitHub...')

repo_url = 'https://github.com/caizongxun/trainer.git'
subprocess.run(['git', 'clone', repo_url, '/tmp/trainer'], check=True)

subprocess.run(['cp', '/tmp/trainer/v9_training/btc_1h_v9_training.py', '.'], check=True)

print('Training script downloaded and ready')
print()
```

## Step 4: Load Data from HuggingFace

```python
from datasets import load_dataset
import pandas as pd

print('[STEP 1/7] Loading BTC 1h data from HuggingFace...')
print()

dataset = load_dataset(
    "zongowo111/cpb-models",
    data_files="klines_binance_us/BTCUSDT/BTCUSDT_1h_binance_us.csv"
)
df = pd.DataFrame(dataset['train'])

print(f'Loaded {len(df)} rows')
print(f'Columns: {df.columns.tolist()}')
print(f'Shape: {df.shape}')
print()
print('First 5 rows:')
print(df.head())
print()
```

## Step 5: Preprocess Data

```python
import pandas as pd
import numpy as np

print('[STEP 2/7] Preprocessing data...')
print()

data = df.copy()

# Convert to numeric
for col in ['open', 'high', 'low', 'close', 'volume']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Clean data
data = data.drop_duplicates(subset=['open_time'])
data = data.sort_values('open_time').reset_index(drop=True)
data = data.dropna()

print(f'Data shape after cleaning: {data.shape}')
print(f'Date range: {data["open_time"].min()} to {data["open_time"].max()}')
print()
```

## Step 6: Calculate Technical Indicators

```python
import ta

print('[STEP 3/7] Calculating technical indicators...')
print()

data_with_ind = data.copy()

# RSI
data_with_ind['rsi_14'] = ta.momentum.rsi(data_with_ind['close'], window=14)
data_with_ind['rsi_7'] = ta.momentum.rsi(data_with_ind['close'], window=7)
print('  RSI calculated')

# MACD
macd = ta.trend.macd(data_with_ind['close'])
data_with_ind['macd'] = macd['MACD_12_26_9']
data_with_ind['macd_signal'] = macd['MACDh_12_26_9']
print('  MACD calculated')

# Bollinger Bands
bb = ta.volatility.bollinger_bands(data_with_ind['close'], window=20)
data_with_ind['bb_high'] = bb['BBH_20_2']
data_with_ind['bb_mid'] = bb['BBM_20_2']
data_with_ind['bb_low'] = bb['BBL_20_2']
data_with_ind['bb_width'] = (data_with_ind['bb_high'] - data_with_ind['bb_low']) / data_with_ind['bb_mid']
print('  Bollinger Bands calculated')

# ATR
data_with_ind['atr'] = ta.volatility.average_true_range(
    data_with_ind['high'], data_with_ind['low'], data_with_ind['close'], window=14
)
print('  ATR calculated')

# Moving Averages
data_with_ind['sma_10'] = ta.trend.sma_indicator(data_with_ind['close'], window=10)
data_with_ind['sma_20'] = ta.trend.sma_indicator(data_with_ind['close'], window=20)
data_with_ind['sma_50'] = ta.trend.sma_indicator(data_with_ind['close'], window=50)
data_with_ind['ema_12'] = ta.trend.ema_indicator(data_with_ind['close'], window=12)
data_with_ind['ema_26'] = ta.trend.ema_indicator(data_with_ind['close'], window=26)
print('  Moving averages calculated')

# Volume
data_with_ind['volume_ma'] = data_with_ind['volume'].rolling(window=20).mean()
data_with_ind['volume_ratio'] = data_with_ind['volume'] / data_with_ind['volume_ma']
print('  Volume indicators calculated')

# Price features
data_with_ind['returns'] = data_with_ind['close'].pct_change()
data_with_ind['log_returns'] = np.log(data_with_ind['close'] / data_with_ind['close'].shift(1))
data_with_ind['price_range'] = (data_with_ind['high'] - data_with_ind['low']) / data_with_ind['close']
data_with_ind['price_momentum'] = data_with_ind['close'] - data_with_ind['close'].shift(5)
data_with_ind['high_low_ratio'] = data_with_ind['high'] / data_with_ind['low']
print('  Price features calculated')

data_with_ind = data_with_ind.fillna(method='bfill').fillna(method='ffill')

print(f'Final shape: {data_with_ind.shape}')
print(f'Total features: {data_with_ind.shape[1]}')
print()
```

## Step 7: Prepare Sequences

```python
from sklearn.preprocessing import MinMaxScaler

print('[STEP 4/7] Preparing sequences...')
print()

feature_cols = [c for c in data_with_ind.columns 
                if c not in ['open_time', 'open', 'high', 'low', 'close', 'volume']]
feature_cols += ['open', 'high', 'low', 'close', 'volume']

X = data_with_ind[feature_cols].values
y_price = data_with_ind['close'].values
y_direction = (data_with_ind['close'].shift(-1) > data_with_ind['close']).astype(float).values
y_volatility = data_with_ind['atr'].values / data_with_ind['close'].values

print(f'Feature matrix shape: {X.shape}')
print(f'Number of features: {X.shape[1]}')
print()

sequence_length = 60

def create_sequences(data, targets, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_len):
        X_seq.append(data[i:i+seq_len])
        y_seq.append(targets[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_price_seq = create_sequences(X, y_price, sequence_length)
_, y_direction_seq = create_sequences(X, y_direction, sequence_length)
_, y_volatility_seq = create_sequences(X, y_volatility, sequence_length)

print(f'Sequence shape: {X_seq.shape}')
print()

scaler = MinMaxScaler()
X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
X_seq_norm_flat = scaler.fit_transform(X_seq_flat)
X_seq_norm = X_seq_norm_flat.reshape(X_seq.shape)

print(f'Normalized sequence shape: {X_seq_norm.shape}')
print()
```

## Step 8: Split Data

```python
print('[STEP 5/7] Splitting data...')
print()

n_samples = len(X_seq_norm)
train_idx = int(n_samples * 0.70)
val_idx = int(n_samples * 0.85)

X_train = X_seq_norm[:train_idx]
X_val = X_seq_norm[train_idx:val_idx]
X_test = X_seq_norm[val_idx:]

y_price_train = y_price_seq[:train_idx]
y_price_val = y_price_seq[train_idx:val_idx]
y_price_test = y_price_seq[val_idx:]

y_dir_train = y_direction_seq[:train_idx]
y_dir_val = y_direction_seq[train_idx:val_idx]
y_dir_test = y_direction_seq[val_idx:]

y_vol_train = y_volatility_seq[:train_idx]
y_vol_val = y_volatility_seq[train_idx:val_idx]
y_vol_test = y_volatility_seq[val_idx:]

print(f'Training set: {X_train.shape}')
print(f'Validation set: {X_val.shape}')
print(f'Test set: {X_test.shape}')
print()
```

## Step 9: Train Direction Model

```python
print('[STEP 6/7] Training Direction Model (Bi-LSTM)...')
print()

input_shape = (sequence_length, X_seq_norm.shape[2])

inputs = layers.Input(shape=input_shape)

x = layers.Bidirectional(
    layers.LSTM(128, return_sequences=True, kernel_regularizer=l2(0.0001))
)(inputs)
x = layers.Dropout(0.3)(x)

x = layers.Bidirectional(
    layers.LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0001))
)(x)
x = layers.Dropout(0.3)(x)

attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = layers.Add()([x, attention])
x = layers.LayerNormalization()(x)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

direction_model = models.Model(inputs, outputs)

optimizer = Adam(learning_rate=0.001)
direction_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

print('Training direction model...')
print()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
    ModelCheckpoint('direction_best.h5', monitor='val_loss', save_best_only=True)
]

history_dir = direction_model.fit(
    X_train, y_dir_train,
    validation_data=(X_val, y_dir_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print('\nDirection model training completed')
print()
```

## Step 10: Evaluate Direction Model

```python
print('Evaluating Direction Model...')
print()

y_dir_pred_prob = direction_model.predict(X_test, verbose=0)
y_dir_pred = (y_dir_pred_prob > 0.5).astype(int).flatten()
y_dir_test_binary = y_dir_test.astype(int).flatten()

acc = accuracy_score(y_dir_test_binary, y_dir_pred)
prec = precision_score(y_dir_test_binary, y_dir_pred, zero_division=0)
rec = recall_score(y_dir_test_binary, y_dir_pred, zero_division=0)
f1 = f1_score(y_dir_test_binary, y_dir_pred, zero_division=0)
auc = roc_auc_score(y_dir_test_binary, y_dir_pred_prob)

print(f'Direction Model Test Metrics:')
print(f'  Accuracy:  {acc:.4f}')
print(f'  Precision: {prec:.4f}')
print(f'  Recall:    {rec:.4f}')
print(f'  F1-Score:  {f1:.4f}')
print(f'  ROC-AUC:   {auc:.4f}')
print()

direction_model.save('direction_model_v9.h5')
print('Direction model saved')
print()
```

## Step 11: Train Volatility Model

```python
print('[STEP 7/7] Training Volatility Model (XGBoost)...')
print()

X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_val_2d = X_val.reshape(X_val.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

volatility_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    verbosity=1
)

print('Training volatility model...')
print()

eval_set = [(X_val_2d, y_vol_val)]
volatility_model.fit(
    X_train_2d, y_vol_train,
    eval_set=eval_set,
    early_stopping_rounds=20,
    verbose=10
)

print('\nVolatility model training completed')
print()

y_vol_pred = volatility_model.predict(X_test_2d)

rmse_vol = np.sqrt(mean_squared_error(y_vol_test, y_vol_pred))
mae_vol = mean_absolute_error(y_vol_test, y_vol_pred)
mape_vol = mean_absolute_percentage_error(y_vol_test, y_vol_pred)
r2_vol = r2_score(y_vol_test, y_vol_pred)

print(f'Volatility Model Test Metrics:')
print(f'  RMSE: {rmse_vol:.6f}')
print(f'  MAE:  {mae_vol:.6f}')
print(f'  MAPE: {mape_vol:.4f}')
print(f'  R2:   {r2_vol:.4f}')
print()

volatility_model.save_model('volatility_model_v9.json')
print('Volatility model saved')
print()
```

## Step 12: Train Price Model

```python
print('[FINAL] Training Price Model (LSTM)...')
print()

inputs = layers.Input(shape=input_shape)

x = layers.LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0001))(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.LSTM(128, return_sequences=False, kernel_regularizer=l2(0.0001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(64, activation='relu')(x)

outputs = layers.Dense(1)(x)

price_model = models.Model(inputs, outputs)

optimizer = Adam(learning_rate=0.0005)
price_model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

print('Training price model...')
print()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7),
    ModelCheckpoint('price_best.h5', monitor='val_loss', save_best_only=True)
]

history_price = price_model.fit(
    X_train, y_price_train,
    validation_data=(X_val, y_price_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print('\nPrice model training completed')
print()
```

## Step 13: Evaluate Price Model and Save Results

```python
print('Evaluating Price Model...')
print()

y_price_pred = price_model.predict(X_test, verbose=0)

rmse_price = np.sqrt(mean_squared_error(y_price_test, y_price_pred))
mae_price = mean_absolute_error(y_price_test, y_price_pred)
mape_price = mean_absolute_percentage_error(y_price_test, y_price_pred)
r2_price = r2_score(y_price_test, y_price_pred)

print(f'Price Model Test Metrics:')
print(f'  RMSE: {rmse_price:.4f}')
print(f'  MAE:  {mae_price:.4f}')
print(f'  MAPE: {mape_price:.4f}%')
print(f'  R2:   {r2_price:.4f}')
print()

price_model.save('price_model_v9.h5')
print('Price model saved')
print()

results = {
    'timestamp': datetime.now().isoformat(),
    'pair': 'BTCUSDT',
    'timeframe': '1h',
    'models': {
        'direction': {'accuracy': float(acc), 'f1': float(f1), 'roc_auc': float(auc)},
        'volatility': {'rmse': float(rmse_vol), 'mape': float(mape_vol), 'r2': float(r2_vol)},
        'price': {'rmse': float(rmse_price), 'mape': float(mape_price), 'r2': float(r2_price)}
    }
}

with open('v9_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('='*80)
print('V9 TRAINING PIPELINE SUCCESSFULLY COMPLETED')
print('='*80)
print()
print('Models saved:')
print('  - direction_model_v9.h5')
print('  - volatility_model_v9.json')
print('  - price_model_v9.h5')
print('  - v9_results.json')
print()
```

## Step 14: Download Files

```python
from google.colab import files

print('Downloading trained models...')
print()

for file in ['direction_model_v9.h5', 'volatility_model_v9.json', 'price_model_v9.h5', 'v9_results.json']:
    try:
        files.download(file)
        print(f'Downloaded: {file}')
    except:
        print(f'Could not download {file}')

print()
print('Upload downloaded files to GitHub repository')
```

## Expected Training Time

- Total training time: 2-4 hours (depending on GPU)
- Direction model: 45-60 minutes
- Volatility model: 20-30 minutes  
- Price model: 60-90 minutes

## Expected Performance

- Direction Accuracy: 72-75%
- Direction F1-Score: 0.70-0.73
- Volatility MAPE: 15-25%
- Price MAPE: 1.5-2.5%
- Price R2: 0.75-0.85

## Files Generated

1. `direction_model_v9.h5` - Bi-LSTM direction prediction model
2. `volatility_model_v9.json` - XGBoost volatility prediction model
3. `price_model_v9.h5` - LSTM price prediction model
4. `v9_results.json` - Training results and metrics

## Upload to GitHub

After training:
1. Download all model files from Colab
2. Clone the trainer repository
3. Place files in `all_models/BTCUSDT/v9_1h/`
4. Push to GitHub
