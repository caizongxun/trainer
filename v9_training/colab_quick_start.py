# V9 BTC 1h Training - Google Colab Quick Start
# Copy and paste this entire script into a Colab cell and run

print('[COLAB] Starting V9 BTC 1h Training Pipeline')
print('[COLAB] This will train 3 specialized models: Direction, Volatility, Price')
print()

print('[COLAB] Step 0: Installing packages...')
import subprocess
import sys

packages = [
    'tensorflow>=2.13.0',
    'xgboost>=2.0.0',
    'datasets>=2.14.0',
    'ta>=0.10.2',
    'pandas>=1.5.0',
    'numpy>=1.23.0',
    'scikit-learn>=1.0.0'
]

for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print('[COLAB] All packages installed')
print()

print('[COLAB] Verifying GPU availability...')
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'[COLAB] GPU devices: {len(gpus)}')
for gpu in gpus:
    print(f'  - {gpu}')
print()

print('[COLAB] Cloning training repository...')
subprocess.run(['git', 'clone', 'https://github.com/caizongxun/trainer.git', '/tmp/trainer'], 
               capture_output=True)
subprocess.run(['cp', '/tmp/trainer/v9_training/btc_1h_v9_training.py', '.'], 
               capture_output=True)

print('[COLAB] Repository ready')
print()

print('[COLAB] Launching training pipeline...')
print('='*80)
print()

from btc_1h_v9_training import V9TrainingPipeline

pipeline = V9TrainingPipeline()
pipeline.run()

print('[COLAB] Training completed')
print('[COLAB] Downloading model files...')
print()

from google.colab import files
import os

for file in ['direction_model_v9.h5', 'volatility_model_v9.json', 'price_model_v9.h5', 'v9_results.json', 'training.log']:
    if os.path.exists(file):
        print(f'[COLAB] Downloading {file}...')
        files.download(file)

print('[COLAB] All files downloaded')
print()
print('[COLAB] Next steps:')
print('1. Upload downloaded files to your GitHub repository')
print('2. Save models to: all_models/BTCUSDT/v9_1h/')
print('3. Review v9_results.json for performance metrics')
print('4. Check training.log for detailed debug information')
