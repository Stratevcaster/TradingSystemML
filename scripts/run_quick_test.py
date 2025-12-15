"""Quick test that loads the small model created by `run_quick_train.py` and performs a prediction
on synthetic data to exercise the test/predict flow without network dependencies.

Usage:
  python scripts/run_quick_test.py
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import parameters
from stock_prediction import load_data
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

model_path = os.path.join('results', 'quicktest.h5')
if not os.path.isfile(model_path):
    raise SystemExit(f"Model file not found: {model_path}. Run scripts/run_quick_train.py first.")

print('Preparing synthetic data for prediction...')
N = max(300, parameters.N_STEPS * 3)
dates = pd.date_range(end=pd.Timestamp.today(), periods=N)
df = pd.DataFrame(index=dates)
np.random.seed(123)
df['adjclose'] = np.cumsum(np.random.normal(0, 1, size=N)) + 100
df['volume'] = np.random.randint(1000, 10000, size=N)
df['open'] = df['adjclose'] + np.random.normal(0, 1, size=N)
df['high'] = df[['open', 'adjclose']].max(axis=1) + np.random.rand(N)
df['low'] = df[['open', 'adjclose']].min(axis=1) - np.random.rand(N)
for col in ['macd', 'atr', 'dma']:
    df[col] = 0.0

data = load_data(df, n_steps=parameters.N_STEPS, shuffle=False, n_days=1, test_size=0.2, feature_columns=parameters.COLUMN_NAME)

model = load_model(model_path, compile=False)

last_sequence = data['last_sequence'][:parameters.N_STEPS]
# reshape and expand dims as the model expects
last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
last_sequence = np.expand_dims(last_sequence, axis=0)

prediction = model.predict(last_sequence)
# Ensure prediction is 2D for inverse_transform
pred = prediction
if getattr(pred, 'ndim', 0) > 2:
  pred = pred.reshape(pred.shape[0], -1)
if getattr(pred, 'ndim', 0) == 1:
  pred = pred.reshape(-1, 1)
predicted_price = data['column_scaler']['adjclose'].inverse_transform(pred)[0][0]
print(f'Predicted price (synthetic) : {predicted_price:.4f}')
