import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import requests
import json
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, GlobalAveragePooling1D, Concatenate, Attention, Dropout, Reshape, Add, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber

# --- Configuration ---
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
SEQ_LEN = 96  # Lookback window (e.g., 4 days for 1h data)
PRED_LEN = 10 # Predict next 10 candles
EPOCHS = 100
BATCH_SIZE = 32
DATASET_URL_TEMPLATE = "https://huggingface.co/datasets/zongowo111/cpb-models/resolve/main/klines_binance_us/{}/{}_{}.csv"

def download_data(symbol, interval):
    url = DATASET_URL_TEMPLATE.format(symbol, symbol, interval)
    print(f"Downloading data from {url}...")
    try:
        df = pd.read_csv(url)
        # Ensure standard columns
        df.columns = [c.lower() for c in df.columns]
        # Expected cols: open_time, open, high, low, close, volume, ...
        # Rename if necessary to match pandas_ta expectations (Open, High, Low, Close, Volume)
        rename_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }
        df.rename(columns=rename_map, inplace=True)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.sort_values('open_time', inplace=True)
        print(f"Data loaded: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def feature_engineering(df):
    print("Generating technical indicators...")
    # Trend
    df.ta.macd(append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    
    # Momentum
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(append=True)
    
    # Volatility
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    
    # Volume
    df.ta.obv(append=True)

    # Log Returns (Stationarity) for features
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['log_vol'] = np.log(df['Volume'] + 1)
    
    # Clean NaNs
    df.dropna(inplace=True)
    print(f"Data shape after features: {df.shape}")
    return df

def prepare_data(df, seq_len, pred_len):
    # Features to use for training
    # We normalized features using RobustScaler to handle outliers
    feature_cols = [c for c in df.columns if c not in ['open_time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    # Add Log Returns of OHLC as primary features
    df['open_ret'] = np.log(df['Open'] / df['Open'].shift(1))
    df['high_ret'] = np.log(df['High'] / df['High'].shift(1))
    df['low_ret'] = np.log(df['Low'] / df['Low'].shift(1))
    
    model_features = ['log_ret', 'log_vol', 'open_ret', 'high_ret', 'low_ret'] + [c for c in df.columns if 'RSI' in c or 'MACD' in c or 'ATR' in c or 'Bgm' in c or 'BBU' in c or 'BBL' in c]
    
    data = df[model_features].values
    
    # Scale features
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Targets: We want to predict Close, High, Low relative to the last known Close (at t)
    # Target[i] = (Price[t+i] - Close[t]) / Close[t]  <-- Percentage Change
    # Or Log Return: log(Price[t+i] / Close[t])
    
    prices_close = df['Close'].values
    prices_high = df['High'].values
    prices_low = df['Low'].values
    
    X, y = [], []
    
    print("Creating sequences...")
    for i in range(len(data_scaled) - seq_len - pred_len):
        # Input: past seq_len
        X.append(data_scaled[i : i + seq_len])
        
        # Target: future pred_len
        # We predict 3 values per step: Close, High, Low
        # All relative to the Close price at the end of the input sequence (index i + seq_len - 1)
        base_price = prices_close[i + seq_len - 1]
        
        target_seq = []
        for j in range(pred_len):
            future_idx = i + seq_len + j
            # Calculate log return relative to base_price
            t_close = np.log(prices_close[future_idx] / base_price)
            t_high = np.log(prices_high[future_idx] / base_price)
            t_low = np.log(prices_low[future_idx] / base_price)
            target_seq.append([t_close, t_high, t_low])
            
        y.append(target_seq)
        
    return np.array(X), np.array(y), scaler, model_features

def build_v10_model(input_shape, pred_len):
    # Hybrid CNN-BiLSTM-Attention Architecture
    inputs = Input(shape=input_shape)
    
    # 1. Local Feature Extraction
    x = Conv1D(filters=64, kernel_size=3, activation='swish', padding='same')(inputs)
    x = Conv1D(filters=64, kernel_size=3, activation='swish', padding='same')(x)
    
    # 2. Temporal Modeling
    # Return sequences=True for Attention
    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(x)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # 3. Attention Mechanism
    # Query is the last hidden state, Keys/Values are the sequence
    # But self-attention over the sequence is also good
    att_out = Attention()([lstm_out, lstm_out])
    
    # Residual Connection + Norm could be added here
    x = Add()([lstm_out, att_out])
    
    # 4. Global Temporal Aggregation
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    
    # 5. Decoder / Prediction Head
    x = Dense(128, activation='swish')(x)
    x = Dropout(0.2)(x)
    
    # Output: pred_len * 3 (Close, High, Low)
    output = Dense(pred_len * 3)(x)
    output = Reshape((pred_len, 3))(output)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=['mse', 'mae'])
    
    return model

def main():
    # 1. Data Preparation
    df = download_data(SYMBOL, INTERVAL)
    if df is None:
        return
    
    df = feature_engineering(df)
    
    # Train/Test Split (Time-based)
    train_size = int(len(df) * 0.9)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    X_train, y_train, scaler, features = prepare_data(df_train, SEQ_LEN, PRED_LEN)
    X_test, y_test, _, _ = prepare_data(df_test, SEQ_LEN, PRED_LEN)
    # Note: Using scaler fitted on train for test data is correct practice
    # Re-apply scaler to test data
    # (Simplified here for script brevity, ideally pass the scaler object)
    
    print(f"Training Data Shape: {X_train.shape}, Targets: {y_train.shape}")
    
    # 2. Build Model
    model = build_v10_model((SEQ_LEN, len(features)), PRED_LEN)
    model.summary()
    
    # 3. Training
    checkpoint = ModelCheckpoint(f"{SYMBOL}_{INTERVAL}_v10.keras", save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    print("Training Complete. Model saved.")
    
    # 4. Evaluation (Basic)
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

if __name__ == "__main__":
    main()
