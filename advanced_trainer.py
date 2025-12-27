#!/usr/bin/env python3
"""
進階虛擬貨幣價格預測訓練模組
支援:
1. 多層LSTM + Attention機制
2. 指標預測 (RSI, MACD, Bollinger Bands)
3. 模型集成
4. 動態超參數調整
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Concatenate, 
    Attention, Layer, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ======================== 技術指標計算 ========================
class TechnicalIndicators:
    """技術指標計算器"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """相對強度指數 (RSI)"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """MACD指標"""
        ema_fast = pd.Series(prices).ewm(span=fast).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow).mean().values
        macd = ema_fast - ema_slow
        signal_line = pd.Series(macd).ewm(span=signal).mean().values
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, num_std=2):
        """布林橘根涨跌帶"""
        sma = pd.Series(prices).rolling(window=period).mean().values
        std = pd.Series(prices).rolling(window=period).std().values
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band

# ======================== 進階LSTM模型 ========================
class AttentionLayer(Layer):
    """自定義Attention層"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)

def create_advanced_model(lookback=60, future_steps=10, use_indicators=True):
    """
    建立進階模型
    
    Args:
        lookback: 回看時間步
        future_steps: 預測步數 (10根K棒)
        use_indicators: 是否使用技術指標
    
    Returns:
        訓練好的Keras模型
    """
    
    if use_indicators:
        # 多輸入模型：OHLC + 指標
        # 輸入1: OHLC價格序列
        price_input = Input(shape=(lookback, 4), name='price_input')
        
        # 輸入2: 技術指標
        indicator_input = Input(shape=(lookback, 5), name='indicator_input')  # RSI, MACD, Signal, Histogram, BB位置
        
        # 價格處理分支
        price_lstm1 = LSTM(128, activation='relu', return_sequences=True)(price_input)
        price_drop1 = Dropout(0.2)(price_lstm1)
        price_lstm2 = LSTM(64, activation='relu', return_sequences=True)(price_drop1)
        price_attn = AttentionLayer()(price_lstm2)
        
        # 指標處理分支
        indicator_lstm = LSTM(64, activation='relu', return_sequences=True)(indicator_input)
        indicator_drop = Dropout(0.2)(indicator_lstm)
        indicator_attn = AttentionLayer()(indicator_drop)
        
        # 合併兩個分支
        merged = Concatenate()([price_attn, indicator_attn])
        dense1 = Dense(128, activation='relu')(merged)
        dense1_drop = Dropout(0.2)(dense1)
        dense2 = Dense(64, activation='relu')(dense1_drop)
        dense2_drop = Dropout(0.2)(dense2)
        
        # 輸出層: 預測10根K棒的OHLC (40個值)
        output = Dense(future_steps * 4, name='price_output')(dense2_drop)
        
        model = Model(inputs=[price_input, indicator_input], outputs=output)
        
    else:
        # 單輸入模型：僅OHLC
        input_layer = Input(shape=(lookback, 4))
        
        lstm1 = LSTM(128, activation='relu', return_sequences=True)(input_layer)
        drop1 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(64, activation='relu', return_sequences=True)(drop1)
        drop2 = Dropout(0.2)(lstm2)
        lstm3 = LSTM(32, activation='relu')(drop2)
        drop3 = Dropout(0.2)(lstm3)
        
        dense1 = Dense(64, activation='relu')(drop3)
        dense1_drop = Dropout(0.2)(dense1)
        output = Dense(future_steps * 4)(dense1_drop)
        
        model = Model(inputs=input_layer, outputs=output)
    
    # 編譯模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ======================== 資料準備函式 ========================
def prepare_advanced_sequences(df, lookback=60, future_steps=10, use_indicators=True):
    """
    準備序列資料 (含指標)
    """
    
    price_data = df[['open', 'high', 'low', 'close']].values
    
    X_price = []
    X_indicators = [] if use_indicators else None
    y = []
    
    for i in range(len(df) - lookback - future_steps + 1):
        # 價格序列
        X_price.append(price_data[i:i+lookback])
        
        # 指標序列
        if use_indicators:
            rsi = df['rsi'].values[i:i+lookback]
            macd = df['macd'].values[i:i+lookback]
            macd_signal = df['macd_signal'].values[i:i+lookback]
            macd_hist = df['macd_hist'].values[i:i+lookback]
            bb_position = df['bb_position'].values[i:i+lookback]
            
            indicators = np.column_stack([
                rsi, macd, macd_signal, macd_hist, bb_position
            ])
            X_indicators.append(indicators)
        
        # 目標輸出 (未來10根K棒的OHLC)
        y.append(price_data[i+lookback:i+lookback+future_steps].flatten())
    
    X_price = np.array(X_price)
    y = np.array(y)
    
    if use_indicators:
        X_indicators = np.array(X_indicators)
        return [X_price, X_indicators], y
    else:
        return X_price, y

def add_technical_indicators(df):
    """
    為DataFrame添加技術指標
    """
    
    indicators = TechnicalIndicators()
    
    close_prices = df['close'].values
    
    # 計算指標
    df['rsi'] = indicators.calculate_rsi(close_prices, period=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = indicators.calculate_macd(close_prices)
    upper_bb, mid_bb, lower_bb = indicators.calculate_bollinger_bands(close_prices)
    
    # BB位置 (0-1之間)
    df['bb_position'] = np.where(
        upper_bb == lower_bb,
        0.5,
        (close_prices - lower_bb) / (upper_bb - lower_bb)
    )
    
    # 填充NaN值
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

# ======================== 訓練函式 ========================
def train_advanced_model(model, X_train, y_train, X_val, y_val, 
                        epochs=100, batch_size=32, patience=15):
    """
    訓練模型 (含早停和學習率調整)
    """
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    評估模型效能
    """
    
    y_pred = model.predict(X_test, verbose=0)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae)
    }
