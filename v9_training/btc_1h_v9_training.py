#!/usr/bin/env python3
"""
BTC 1h v9 Multi-Task Cryptocurrency Price Prediction Model
Specialized models for direction, volatility, and price prediction
Training script for Google Colab with GPU optimization
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import json
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            mean_absolute_percentage_error, r2_score,
                            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

import ta
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CryptoDataLoader:
    """Load and preprocess cryptocurrency OHLCV data"""
    
    def __init__(self, pair='BTCUSDT', timeframe='1h'):
        self.pair = pair
        self.timeframe = timeframe
        self.df = None
        self.logger = logging.getLogger(__name__)
        
    def load_from_huggingface(self):
        """Load data from HuggingFace dataset"""
        self.logger.info(f'Loading {self.pair} {self.timeframe} data from HuggingFace...')
        try:
            dataset = load_dataset(
                "zongowo111/cpb-models",
                data_files=f"klines_binance_us/{self.pair}/{self.pair}_{self.timeframe}_binance_us.csv"
            )
            df = pd.DataFrame(dataset['train'])
            self.logger.info(f'Successfully loaded {len(df)} rows')
            self.df = df
            return df
        except Exception as e:
            self.logger.error(f'Error loading data: {e}')
            raise
    
    def preprocess_data(self):
        """Clean and prepare data"""
        if self.df is None:
            raise ValueError('Data not loaded. Call load_from_huggingface first.')
        
        self.logger.info('Preprocessing data...')
        
        df = self.df.copy()
        
        # Ensure correct column names
        column_mapping = {
            'open_time': 'open_time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        for old, new in column_mapping.items():
            if old in df.columns and old != new:
                df.rename(columns={old: new}, inplace=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['open_time'])
        df = df.sort_values('open_time').reset_index(drop=True)
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN
        df = df.dropna()
        
        self.logger.info(f'Data shape after cleaning: {df.shape}')
        self.df = df
        return df

class TechnicalIndicatorCalculator:
    """Calculate technical indicators for feature engineering"""
    
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate comprehensive technical indicators"""
        logger.info('Calculating technical indicators...')
        
        df = df.copy()
        
        # Momentum indicators
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
        
        # MACD
        macd = ta.trend.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bb = ta.volatility.bollinger_bands(df['close'], window=20)
        df['bb_high'] = bb['BBH_20_2']
        df['bb_mid'] = bb['BBM_20_2']
        df['bb_low'] = bb['BBL_20_2']
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # Moving averages
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['price_momentum'] = df['close'] - df['close'].shift(5)
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f'Added {len(df.columns)} features including {len([c for c in df.columns if "rsi" in c or "macd" in c or "bb" in c or "atr" in c])} technical indicators')
        return df

class DataSequenceGenerator:
    """Generate sequences for LSTM training"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def create_sequences(self, X, y=None):
        """Create windowed sequences"""
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            if y is not None:
                y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq) if y is not None else None
    
    def normalize_data(self, X, fit=True):
        """Normalize sequences"""
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        
        if fit:
            X_norm = self.scaler_x.fit_transform(X_flat)
        else:
            X_norm = self.scaler_x.transform(X_flat)
        
        return X_norm.reshape(original_shape)

class DirectionPredictionModel:
    """Bi-LSTM with attention for direction prediction (UP/DOWN)"""
    
    def __init__(self, input_shape, attention_units=64, lstm_units=128, dropout=0.3):
        self.model = self._build_model(input_shape, attention_units, lstm_units, dropout)
        self.logger = logging.getLogger(__name__)
        
    def _build_model(self, input_shape, attention_units, lstm_units, dropout):
        """Build Bi-LSTM with attention architecture"""
        
        inputs = layers.Input(shape=input_shape)
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, 
                       kernel_regularizer=l2(0.0001))
        )(inputs)
        x = layers.Dropout(dropout)(x)
        
        x = layers.Bidirectional(
            layers.LSTM(lstm_units // 2, return_sequences=True,
                       kernel_regularizer=l2(0.0001))
        )(x)
        x = layers.Dropout(dropout)(x)
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=attention_units
        )(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        
        # Output layer (binary classification)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def compile(self, learning_rate=0.001):
        """Compile model"""
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        self.logger.info('Direction model compiled')
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train model"""
        self.logger.info('Training direction model...')
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint('direction_best.h5', monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info('Direction model training completed')
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_test_binary = y_test.astype(int).flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_test_binary, y_pred),
            'precision': precision_score(y_test_binary, y_pred, zero_division=0),
            'recall': recall_score(y_test_binary, y_pred, zero_division=0),
            'f1': f1_score(y_test_binary, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test_binary, y_pred_prob),
        }
        
        self.logger.info(f'Direction Model Metrics: {metrics}')
        return metrics, y_pred_prob

class VolatilityPredictionModel:
    """XGBoost model for volatility prediction"""
    
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.05):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            verbosity=1
        )
        self.logger = logging.getLogger(__name__)
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        self.logger.info('Training volatility model...')
        
        eval_set = [(X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=20,
            verbose=10
        )
        
        self.logger.info('Volatility model training completed')
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
        }
        
        self.logger.info(f'Volatility Model Metrics: {metrics}')
        return metrics, y_pred

class PricePredictionModel:
    """LSTM for price level prediction"""
    
    def __init__(self, input_shape, lstm_units=256, dropout=0.3):
        self.model = self._build_model(input_shape, lstm_units, dropout)
        self.logger = logging.getLogger(__name__)
        
    def _build_model(self, input_shape, lstm_units, dropout):
        """Build LSTM architecture"""
        
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        x = layers.LSTM(lstm_units, return_sequences=True,
                       kernel_regularizer=l2(0.0001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        x = layers.LSTM(lstm_units // 2, return_sequences=False,
                       kernel_regularizer=l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        
        # Output
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def compile(self, learning_rate=0.0005):
        """Compile model"""
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        self.logger.info('Price model compiled')
    
    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32):
        """Train model"""
        self.logger.info('Training price model...')
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7),
            ModelCheckpoint('price_best.h5', monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info('Price model training completed')
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        y_pred = self.model.predict(X_test, verbose=0)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
        }
        
        self.logger.info(f'Price Model Metrics: {metrics}')
        return metrics, y_pred

class TrainingPipeline:
    """Main training pipeline"""
    
    def __init__(self, pair='BTCUSDT', timeframe='1h', sequence_length=60):
        self.pair = pair
        self.timeframe = timeframe
        self.sequence_length = sequence_length
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Execute full training pipeline"""
        
        self.logger.info('='*80)
        self.logger.info(f'Starting V9 Model Training for {self.pair} {self.timeframe}')
        self.logger.info('='*80)
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        self.logger.info(f'GPUs available: {len(gpus)}')
        if gpus:
            for gpu in gpus:
                self.logger.info(f'  - {gpu}')
        
        # Step 1: Load data
        self.logger.info('\n[STEP 1/7] Loading data...')
        loader = CryptoDataLoader(self.pair, self.timeframe)
        loader.load_from_huggingface()
        df = loader.preprocess_data()
        
        # Step 2: Calculate indicators
        self.logger.info('\n[STEP 2/7] Calculating technical indicators...')
        df = TechnicalIndicatorCalculator.calculate_all_indicators(df)
        self.logger.info(f'Data shape: {df.shape}')
        
        # Step 3: Prepare sequences and targets
        self.logger.info('\n[STEP 3/7] Preparing sequences...')
        
        feature_cols = [c for c in df.columns if c not in ['open_time', 'open', 'high', 'low', 'close', 'volume']]
        feature_cols += ['open', 'high', 'low', 'close', 'volume']
        
        X = df[feature_cols].values
        y_price = df['close'].values
        
        # Create direction labels (1=up, 0=down)
        y_direction = (df['close'].shift(-1) > df['close']).astype(float).values
        
        # Create volatility labels (normalized ATR)
        y_volatility = df['atr'].values / df['close'].values
        
        # Generate sequences
        seq_gen = DataSequenceGenerator(self.sequence_length)
        X_seq, y_price_seq = seq_gen.create_sequences(X, y_price)
        _, y_direction_seq = seq_gen.create_sequences(X, y_direction)
        _, y_volatility_seq = seq_gen.create_sequences(X, y_volatility)
        
        # Normalize sequences
        X_seq_norm = seq_gen.normalize_data(X_seq, fit=True)
        
        self.logger.info(f'Sequence shape: {X_seq_norm.shape}')
        self.logger.info(f'Price targets shape: {y_price_seq.shape}')
        
        # Step 4: Train/Val/Test split
        self.logger.info('\n[STEP 4/7] Splitting data...')
        
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
        
        self.logger.info(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
        
        # Step 5: Train direction model
        self.logger.info('\n[STEP 5/7] Training direction prediction model...')
        
        dir_model = DirectionPredictionModel(
            input_shape=(self.sequence_length, X_seq_norm.shape[2]),
            attention_units=64,
            lstm_units=128,
            dropout=0.3
        )
        dir_model.compile(learning_rate=0.001)
        dir_history = dir_model.train(X_train, y_dir_train, X_val, y_dir_val, epochs=100, batch_size=32)
        dir_metrics, dir_pred = dir_model.evaluate(X_test, y_dir_test)
        
        # Step 6: Train volatility model
        self.logger.info('\n[STEP 6/7] Training volatility prediction model...')
        
        # For XGBoost, use flattened 2D representation
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_val_2d = X_val.reshape(X_val.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        
        vol_model = VolatilityPredictionModel(n_estimators=300, max_depth=6, learning_rate=0.05)
        vol_model.train(X_train_2d, y_vol_train, X_val_2d, y_vol_val)
        vol_metrics, vol_pred = vol_model.evaluate(X_test_2d, y_vol_test)
        
        # Train price model
        self.logger.info('\n[STEP 7/7] Training price prediction model...')
        
        price_model = PricePredictionModel(
            input_shape=(self.sequence_length, X_seq_norm.shape[2]),
            lstm_units=256,
            dropout=0.3
        )
        price_model.compile(learning_rate=0.0005)
        price_history = price_model.train(X_train, y_price_train, X_val, y_price_val, epochs=150, batch_size=32)
        price_metrics, price_pred = price_model.evaluate(X_test, y_price_test)
        
        # Save results
        self.logger.info('\n[FINAL] Saving results...')
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'pair': self.pair,
            'timeframe': self.timeframe,
            'sequence_length': self.sequence_length,
            'direction_metrics': dir_metrics,
            'volatility_metrics': vol_metrics,
            'price_metrics': price_metrics,
            'data_shapes': {
                'X_train': X_train.shape,
                'X_val': X_val.shape,
                'X_test': X_test.shape,
            }
        }
        
        with open('v9_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save models
        dir_model.model.save('direction_model_v9.h5')
        vol_model.model.save_model('volatility_model_v9.json')
        price_model.model.save('price_model_v9.h5')
        
        self.logger.info('\n' + '='*80)
        self.logger.info('TRAINING COMPLETE')
        self.logger.info('='*80)
        self.logger.info(f'Direction Accuracy: {dir_metrics["accuracy"]:.4f}')
        self.logger.info(f'Volatility RMSE: {vol_metrics["rmse"]:.6f}')
        self.logger.info(f'Price MAPE: {price_metrics["mape"]:.4f}%')
        self.logger.info('='*80)
        
        return results

if __name__ == '__main__':
    pipeline = TrainingPipeline(pair='BTCUSDT', timeframe='1h', sequence_length=60)
    results = pipeline.run()
