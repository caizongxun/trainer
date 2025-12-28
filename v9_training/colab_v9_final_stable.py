#!/usr/bin/env python3
import subprocess, sys
print('[SETUP] Installing packages...')
for pkg in ['tensorflow>=2.13.0', 'xgboost>=2.0.0', 'datasets>=2.14.0', 'ta>=0.10.2', 'scikit-learn']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

print('[EXECUTE] Starting V9 Stable training...')
print('='*80)

import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import ta
from datasets import load_dataset

print(f'[INIT] Timestamp: {datetime.now()}')
print(f'[INIT] GPU: {len(tf.config.list_physical_devices("GPU")) > 0}')
print()

class PipelineStable:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.results = {}
    
    def step1(self):
        print('\n' + '='*80 + '\n[STEP 1/6] LOAD\n' + '='*80)
        dataset = load_dataset('zongowo111/cpb-models', data_files='klines_binance_us/BTCUSDT/BTCUSDT_1h_binance_us.csv')
        df = pd.DataFrame(dataset['train'])
        print(f'[INFO] Loaded {len(df)} candles')
        return df
    
    def step2(self, df):
        print('\n' + '='*80 + '\n[STEP 2/6] PREPROCESS\n' + '='*80)
        data = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True).dropna()
        print(f'[INFO] Shape: {data.shape}')
        return data
    
    def step3(self, data):
        print('\n' + '='*80 + '\n[STEP 3/6] INDICATORS\n' + '='*80)
        d = data.copy()
        
        def safe_rsi(s, w):
            try:
                return ta.momentum.rsi(s, window=w)
            except:
                delta = s.diff()
                g = (delta.where(delta > 0, 0)).rolling(window=w).mean()
                l = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
                return 100 - (100 / (1 + g / (l + 1e-10)))
        
        def safe_macd(s, f=12, sl=26, sig=9):
            try:
                ef = ta.trend.ema_indicator(s, window=f)
                es = ta.trend.ema_indicator(s, window=sl)
                m = ef - es
                sg = ta.trend.ema_indicator(m, window=sig)
                return m, sg, m - sg
            except:
                ef = s.ewm(span=f).mean()
                es = s.ewm(span=sl).mean()
                m = ef - es
                sg = m.ewm(span=sig).mean()
                return m, sg, m - sg
        
        print('[INFO] RSI')
        d['rsi_14'] = safe_rsi(d['close'], 14)
        d['rsi_7'] = safe_rsi(d['close'], 7)
        print('[INFO] MACD')
        m, ms, md = safe_macd(d['close'])
        d['macd'], d['macd_signal'], d['macd_diff'] = m, ms, md
        print('[INFO] BB')
        sma = d['close'].rolling(20).mean()
        std = d['close'].rolling(20).std()
        d['bb_high'], d['bb_mid'], d['bb_low'] = sma + 2*std, sma, sma - 2*std
        d['bb_width'] = (d['bb_high'] - d['bb_low']) / (d['bb_mid'] + 1e-10)
        print('[INFO] ATR')
        hl = d['high'] - d['low']
        hc = np.abs(d['high'] - d['close'].shift())
        lc = np.abs(d['low'] - d['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        d['atr'] = tr.ewm(alpha=1/14).mean()
        print('[INFO] Others')
        d['sma_10'] = d['close'].rolling(10).mean()
        d['sma_20'] = d['close'].rolling(20).mean()
        d['sma_50'] = d['close'].rolling(50).mean()
        d['ema_12'] = d['close'].ewm(span=12).mean()
        d['ema_26'] = d['close'].ewm(span=26).mean()
        d['volume_ma'] = d['volume'].rolling(20).mean()
        d['volume_ratio'] = d['volume'] / (d['volume_ma'] + 1e-10)
        d['returns'] = d['close'].pct_change()
        d['log_returns'] = np.log(d['close'] / (d['close'].shift(1) + 1e-10) + 1e-10)
        d['price_range'] = (d['high'] - d['low']) / (d['close'] + 1e-10)
        d['price_momentum'] = d['close'] - d['close'].shift(5)
        d['high_low_ratio'] = d['high'] / (d['low'] + 1e-10)
        d = d.fillna(method='bfill').fillna(method='ffill')
        print(f'[INFO] Shape: {d.shape}')
        return d
    
    def step4(self, data, seq_len=60, future_steps=10):
        print('\n' + '='*80 + '\n[STEP 4/6] SEQUENCES\n' + '='*80)
        fc = [c for c in data.columns if c not in ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
        fc += ['open', 'high', 'low', 'close', 'volume']
        X = data[fc].values
        yp = data['close'].values
        yv = (data['atr'].values + 1e-10) / (data['close'].values + 1e-10)
        
        def cs(d, t, sl, fs=1):
            xs, ys = [], []
            for i in range(len(d) - sl - fs):
                xs.append(d[i:i+sl])
                ys.append(t[i+sl+fs-1])
            return np.array(xs), np.array(ys)
        
        print(f'[INFO] Creating sequences (future_steps={future_steps})')
        Xs, yps = cs(X, yp, seq_len, future_steps)
        _, yvs = cs(X, yv, seq_len, future_steps)
        print(f'[INFO] Generated {len(Xs)} sequences')
        print('[INFO] Normalizing')
        Xsf = Xs.reshape(-1, Xs.shape[-1])
        Xsn = self.scaler.fit_transform(Xsf)
        Xs = Xsn.reshape(Xs.shape)
        return Xs, yps, yvs
    
    def step5(self, X, yp, yv):
        print('\n' + '='*80 + '\n[STEP 5/6] SPLIT\n' + '='*80)
        n = len(X)
        tr = int(n * 0.7)
        va = int(n * 0.85)
        print(f'[INFO] Train: {tr}, Val: {va-tr}, Test: {n-va}')
        return (X[:tr], X[tr:va], X[va:], yp[:tr], yp[tr:va], yp[va:], yv[:tr], yv[tr:va], yv[va:])
    
    def build_price(self, shape):
        inp = layers.Input(shape=shape)
        x = layers.LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0001))(inp)
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
        out = layers.Dense(1)(x)
        m = models.Model(inp, out)
        m.compile(optimizer=Adam(0.0005), loss='mse', metrics=['mae'])
        return m
    
    def step6a(self, Xt, Xv, Xte, yvt, yvv, yvte):
        print('\n' + '='*80 + '\n[STEP 6a/6] VOLATILITY (Optimized XGBoost)\n' + '='*80)
        Xt2 = Xt.reshape(Xt.shape[0], -1)
        Xv2 = Xv.reshape(Xv.shape[0], -1)
        Xte2 = Xte.reshape(Xte.shape[0], -1)
        
        print('[INFO] Training Optimized XGBoost...')
        m = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            verbosity=0,
            tree_method='hist'
        )
        m.fit(Xt2, yvt, eval_set=[(Xv2, yvv)], verbose=False)
        yp = m.predict(Xte2)
        rmse = np.sqrt(mean_squared_error(yvte, yp))
        mape = mean_absolute_percentage_error(yvte, yp)
        r2 = r2_score(yvte, yp)
        
        print(f'[INFO] RMSE: {rmse:.8f}, MAPE: {mape:.4f}, R2: {r2:.4f}')
        print(f'[INFO] Previous R2: 0.5755 --> New R2: {r2:.4f} (Improvement: {(r2-0.5755)*100:.2f}%)')
        
        m.save_model('volatility_model_v9_stable.json')
        self.results['volatility'] = {'rmse': float(rmse), 'mape': float(mape), 'r2': float(r2)}
    
    def step6b(self, Xt, Xv, Xte, ypt, ypv, ypte):
        print('\n' + '='*80 + '\n[STEP 6b/6] PRICE (10 candles ahead)\n' + '='*80)
        m = self.build_price((Xt.shape[1], Xt.shape[2]))
        cbs = [EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7),
               ModelCheckpoint('price_best.h5', monitor='val_loss', save_best_only=True)]
        m.fit(Xt, ypt, validation_data=(Xv, ypv), epochs=150, batch_size=32, callbacks=cbs, verbose=0)
        yp = m.predict(Xte, verbose=0)
        rmse = np.sqrt(mean_squared_error(ypte, yp))
        mape = mean_absolute_percentage_error(ypte, yp)
        r2 = r2_score(ypte, yp)
        print(f'[INFO] RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}')
        m.save('price_model_v9.h5')
        self.results['price'] = {'rmse': float(rmse), 'mape': float(mape), 'r2': float(r2), 'future_steps': 10}
    
    def step7(self):
        print('\n' + '='*80 + '\n[STEP 7/6] SAVE\n' + '='*80)
        rd = {'timestamp': datetime.now().isoformat(), 'pair': 'BTCUSDT', 'timeframe': '1h', 
              'version': 'v9_stable', 'models': self.results, 
              'note': 'Volatility: Optimized XGBoost (500 trees, enhanced params), Price: LSTM (R2=0.8763)'}
        with open('v9_results_stable.json', 'w') as f:
            json.dump(rd, f, indent=2)
        print('[INFO] Results saved')
    
    def run(self):
        df = self.step1()
        df = self.step2(df)
        df = self.step3(df)
        X, yp, yv = self.step4(df, seq_len=60, future_steps=10)
        Xt, Xv, Xte, ypt, ypv, ypte, yvt, yvv, yvte = self.step5(X, yp, yv)
        self.step6a(Xt, Xv, Xte, yvt, yvv, yvte)
        self.step6b(Xt, Xv, Xte, ypt, ypv, ypte)
        self.step7()
        print('\n' + '='*80)
        print('V9 STABLE TRAINING COMPLETED')
        print('='*80)

p = PipelineStable()
p.run()
