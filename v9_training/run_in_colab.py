import os
import sys
import json
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

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

print('[INIT] V9 BTC 1h Training Pipeline Started')
print(f'[INIT] Timestamp: {datetime.now()}')
print(f'[INIT] GPU Available: {len(tf.config.list_physical_devices("GPU")) > 0}')
print()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class V9TrainingPipeline:
    def __init__(self):
        self.logger = logger
        self.scaler = MinMaxScaler()
        self.results = {}
        
    def log(self, message, level='INFO'):
        if level == 'INFO':
            self.logger.info(message)
        elif level == 'DEBUG':
            self.logger.debug(message)
        elif level == 'ERROR':
            self.logger.error(message)
        print(f'[{level}] {message}')

    def step_1_load_data(self):
        print('\n' + '='*80)
        print('[STEP 1/7] LOADING DATA FROM HUGGINGFACE')
        print('='*80)
        
        self.log('Loading BTC 1h data from HuggingFace dataset')
        
        try:
            dataset = load_dataset(
                'zongowo111/cpb-models',
                data_files='klines_binance_us/BTCUSDT/BTCUSDT_1h_binance_us.csv'
            )
            df = pd.DataFrame(dataset['train'])
            
            self.log(f'Successfully loaded {len(df)} candles')
            self.log(f'Columns: {df.columns.tolist()}')
            
            return df
            
        except Exception as e:
            self.log(f'Error loading: {e}', level='ERROR')
            raise

    def step_2_preprocess_data(self, df):
        print('\n' + '='*80)
        print('[STEP 2/7] PREPROCESSING DATA')
        print('='*80)
        
        self.log('Converting columns to numeric')
        data = df.copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        self.log('Removing duplicates and sorting')
        data = data.drop_duplicates(subset=['open_time'])
        data = data.sort_values('open_time').reset_index(drop=True)
        data = data.dropna()
        
        self.log(f'Data shape: {data.shape}')
        
        return data

    def calculate_rsi(self, data, window=14, col='close'):
        try:
            return ta.momentum.rsi(data[col], window=window)
        except:
            delta = data[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))

    def calculate_macd(self, data, col='close', fast=12, slow=26, signal=9):
        try:
            ema_fast = ta.trend.ema_indicator(data[col], window=fast)
            ema_slow = ta.trend.ema_indicator(data[col], window=slow)
            macd = ema_fast - ema_slow
            signal_line = ta.trend.ema_indicator(macd, window=signal)
            macd_diff = macd - signal_line
            return macd, signal_line, macd_diff
        except:
            ema_fast = data[col].ewm(span=fast).mean()
            ema_slow = data[col].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            macd_diff = macd - signal_line
            return macd, signal_line, macd_diff

    def calculate_bollinger_bands(self, data, window=20, col='close'):
        try:
            sma = ta.trend.sma_indicator(data[col], window=window)
            std = data[col].rolling(window=window).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            return upper, sma, lower
        except:
            sma = data[col].rolling(window=window).mean()
            std = data[col].rolling(window=window).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            return upper, sma, lower

    def calculate_atr(self, data, window=14):
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            return true_range.ewm(alpha=1/window).mean()
        except:
            return (data['high'] - data['low']).rolling(window=window).mean()

    def step_3_calculate_indicators(self, data):
        print('\n' + '='*80)
        print('[STEP 3/7] CALCULATING TECHNICAL INDICATORS')
        print('='*80)
        
        self.log('Starting technical indicator calculation')
        data_ind = data.copy()
        
        try:
            self.log('RSI')
            data_ind['rsi_14'] = self.calculate_rsi(data_ind, window=14)
            data_ind['rsi_7'] = self.calculate_rsi(data_ind, window=7)
            
            self.log('MACD')
            macd, signal, diff = self.calculate_macd(data_ind)
            data_ind['macd'] = macd
            data_ind['macd_signal'] = signal
            data_ind['macd_diff'] = diff
            
            self.log('Bollinger Bands')
            upper, mid, lower = self.calculate_bollinger_bands(data_ind)
            data_ind['bb_high'] = upper
            data_ind['bb_mid'] = mid
            data_ind['bb_low'] = lower
            data_ind['bb_width'] = (upper - lower) / (mid + 1e-10)
            
            self.log('ATR')
            data_ind['atr'] = self.calculate_atr(data_ind)
            
            self.log('Moving Averages')
            data_ind['sma_10'] = data_ind['close'].rolling(window=10).mean()
            data_ind['sma_20'] = data_ind['close'].rolling(window=20).mean()
            data_ind['sma_50'] = data_ind['close'].rolling(window=50).mean()
            data_ind['ema_12'] = data_ind['close'].ewm(span=12).mean()
            data_ind['ema_26'] = data_ind['close'].ewm(span=26).mean()
            
            self.log('Volume indicators')
            data_ind['volume_ma'] = data_ind['volume'].rolling(window=20).mean()
            data_ind['volume_ratio'] = data_ind['volume'] / (data_ind['volume_ma'] + 1e-10)
            
            self.log('Price features')
            data_ind['returns'] = data_ind['close'].pct_change()
            data_ind['log_returns'] = np.log(data_ind['close'] / (data_ind['close'].shift(1) + 1e-10) + 1e-10)
            data_ind['price_range'] = (data_ind['high'] - data_ind['low']) / (data_ind['close'] + 1e-10)
            data_ind['price_momentum'] = data_ind['close'] - data_ind['close'].shift(5)
            data_ind['high_low_ratio'] = data_ind['high'] / (data_ind['low'] + 1e-10)
            
            data_ind = data_ind.fillna(method='bfill').fillna(method='ffill')
            
            self.log(f'All indicators calculated. Shape: {data_ind.shape}')
            return data_ind
            
        except Exception as e:
            self.log(f'Error: {str(e)}', level='ERROR')
            raise

    def step_4_prepare_sequences(self, data, sequence_length=60):
        print('\n' + '='*80)
        print('[STEP 4/7] PREPARING SEQUENCES')
        print('='*80)
        
        self.log(f'Sequence length: {sequence_length}')
        
        feature_cols = [c for c in data.columns 
                       if c not in ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
        feature_cols += ['open', 'high', 'low', 'close', 'volume']
        
        X = data[feature_cols].values
        y_price = data['close'].values
        y_direction = (data['close'].shift(-1) > data['close']).astype(float).values
        y_volatility = (data['atr'].values + 1e-10) / (data['close'].values + 1e-10)
        
        def create_sequences(data, targets, seq_len):
            X_seq, y_seq = [], []
            for i in range(len(data) - seq_len):
                X_seq.append(data[i:i+seq_len])
                y_seq.append(targets[i+seq_len])
            return np.array(X_seq), np.array(y_seq)
        
        self.log('Creating sequences')
        X_seq, y_price_seq = create_sequences(X, y_price, sequence_length)
        _, y_direction_seq = create_sequences(X, y_direction, sequence_length)
        _, y_volatility_seq = create_sequences(X, y_volatility, sequence_length)
        
        self.log(f'Generated {len(X_seq)} sequences')
        
        self.log('Normalizing')
        X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
        X_seq_norm_flat = self.scaler.fit_transform(X_seq_flat)
        X_seq_norm = X_seq_norm_flat.reshape(X_seq.shape)
        
        return X_seq_norm, y_price_seq, y_direction_seq, y_volatility_seq

    def step_5_split_data(self, X, y_price, y_direction, y_volatility):
        print('\n' + '='*80)
        print('[STEP 5/7] SPLITTING TRAIN/VAL/TEST')
        print('='*80)
        
        n_samples = len(X)
        train_idx = int(n_samples * 0.70)
        val_idx = int(n_samples * 0.85)
        
        self.log(f'Train: {train_idx}, Val: {val_idx-train_idx}, Test: {n_samples-val_idx}')
        
        return (X[:train_idx], X[train_idx:val_idx], X[val_idx:],
                y_price[:train_idx], y_price[train_idx:val_idx], y_price[val_idx:],
                y_direction[:train_idx], y_direction[train_idx:val_idx], y_direction[val_idx:],
                y_volatility[:train_idx], y_volatility[train_idx:val_idx], y_volatility[val_idx:])

    def build_direction_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, kernel_regularizer=l2(0.0001)))(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0001)))(x)
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
        model = models.Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
        return model

    def build_price_model(self, input_shape):
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
        model = models.Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
        return model

    def step_6_train_direction_model(self, X_train, X_val, X_test, y_dir_train, y_dir_val, y_dir_test):
        print('\n' + '='*80)
        print('[STEP 6a/7] TRAINING DIRECTION MODEL')
        print('='*80)
        
        model = self.build_direction_model((X_train.shape[1], X_train.shape[2]))
        callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
                    ModelCheckpoint('direction_best.h5', monitor='val_loss', save_best_only=True)]
        model.fit(X_train, y_dir_train, validation_data=(X_val, y_dir_val), epochs=100, batch_size=32, callbacks=callbacks, verbose=1)
        
        y_dir_pred_prob = model.predict(X_test, verbose=0)
        y_dir_pred = (y_dir_pred_prob > 0.5).astype(int).flatten()
        y_dir_test_binary = y_dir_test.astype(int).flatten()
        
        acc = accuracy_score(y_dir_test_binary, y_dir_pred)
        f1 = f1_score(y_dir_test_binary, y_dir_pred, zero_division=0)
        auc = roc_auc_score(y_dir_test_binary, y_dir_pred_prob)
        
        self.log(f'Direction Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
        model.save('direction_model_v9.h5')
        
        self.results['direction'] = {'accuracy': float(acc), 'f1': float(f1), 'auc': float(auc)}

    def step_6b_train_volatility_model(self, X_train, X_val, X_test, y_vol_train, y_vol_val, y_vol_test):
        print('\n' + '='*80)
        print('[STEP 6b/7] TRAINING VOLATILITY MODEL')
        print('='*80)
        
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_val_2d = X_val.reshape(X_val.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        
        model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', random_state=42, verbosity=0)
        model.fit(X_train_2d, y_vol_train, eval_set=[(X_val_2d, y_vol_val)], early_stopping_rounds=20, verbose=0)
        
        y_vol_pred = model.predict(X_test_2d)
        rmse = np.sqrt(mean_squared_error(y_vol_test, y_vol_pred))
        mape = mean_absolute_percentage_error(y_vol_test, y_vol_pred)
        r2 = r2_score(y_vol_test, y_vol_pred)
        
        self.log(f'Volatility RMSE: {rmse:.8f}, MAPE: {mape:.4f}, R2: {r2:.4f}')
        model.save_model('volatility_model_v9.json')
        
        self.results['volatility'] = {'rmse': float(rmse), 'mape': float(mape), 'r2': float(r2)}

    def step_6c_train_price_model(self, X_train, X_val, X_test, y_price_train, y_price_val, y_price_test):
        print('\n' + '='*80)
        print('[STEP 6c/7] TRAINING PRICE MODEL')
        print('='*80)
        
        model = self.build_price_model((X_train.shape[1], X_train.shape[2]))
        callbacks = [EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7),
                    ModelCheckpoint('price_best.h5', monitor='val_loss', save_best_only=True)]
        model.fit(X_train, y_price_train, validation_data=(X_val, y_price_val), epochs=150, batch_size=32, callbacks=callbacks, verbose=1)
        
        y_price_pred = model.predict(X_test, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_price_test, y_price_pred))
        mape = mean_absolute_percentage_error(y_price_test, y_price_pred)
        r2 = r2_score(y_price_test, y_price_pred)
        
        self.log(f'Price RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}')
        model.save('price_model_v9.h5')
        
        self.results['price'] = {'rmse': float(rmse), 'mape': float(mape), 'r2': float(r2)}

    def step_7_save_results(self):
        print('\n' + '='*80)
        print('[STEP 7/7] SAVING RESULTS')
        print('='*80)
        
        results_data = {'timestamp': datetime.now().isoformat(), 'pair': 'BTCUSDT', 'timeframe': '1h', 'version': 'v9', 'models': self.results}
        with open('v9_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        self.log('Results saved')

    def run(self):
        try:
            df = self.step_1_load_data()
            df = self.step_2_preprocess_data(df)
            df = self.step_3_calculate_indicators(df)
            X, y_price, y_direction, y_volatility = self.step_4_prepare_sequences(df)
            splits = self.step_5_split_data(X, y_price, y_direction, y_volatility)
            X_train, X_val, X_test, y_price_train, y_price_val, y_price_test, y_dir_train, y_dir_val, y_dir_test, y_vol_train, y_vol_val, y_vol_test = splits
            self.step_6_train_direction_model(X_train, X_val, X_test, y_dir_train, y_dir_val, y_dir_test)
            self.step_6b_train_volatility_model(X_train, X_val, X_test, y_vol_train, y_vol_val, y_vol_test)
            self.step_6c_train_price_model(X_train, X_val, X_test, y_price_train, y_price_val, y_price_test)
            self.step_7_save_results()
            print('\n' + '='*80)
            print('V9 TRAINING SUCCESSFULLY COMPLETED')
            print('='*80)
        except Exception as e:
            self.log(f'Pipeline failed: {str(e)}', level='ERROR')
            raise

pipeline = V9TrainingPipeline()
pipeline.run()
