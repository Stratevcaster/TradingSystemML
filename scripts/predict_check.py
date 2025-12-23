import time
from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from stock_prediction import create_model, load_data
from parameters import *
import numpy as np
import pandas as pd
import os

# Build synthetic df
N = max(300, N_STEPS * 3)
dates = pd.date_range(end=pd.Timestamp.today(), periods=N)
df = pd.DataFrame(index=dates)
np.random.seed(42)
df['adjclose'] = np.cumsum(np.random.normal(0, 1, size=N)) + 100
df['volume'] = np.random.randint(1000, 10000, size=N)
df['open'] = df['adjclose'] + np.random.normal(0, 1, size=N)
df['high'] = df[['open', 'adjclose']].max(axis=1) + np.random.rand(N)
df['low'] = df[['open', 'adjclose']].min(axis=1) - np.random.rand(N)
for col in ['macd', 'atr', 'dma']:
    df[col] = 0.0

data = load_data(df, n_steps=N_STEPS, shuffle=False, n_days=1, test_size=TEST_SIZE, feature_columns=COLUMN_NAME)
print('X_test shape', data['X_test'].shape)

model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, num_layers=NUM_LAYERS,
                     dropout=DROPOUT, normalizer=normalizer, bidirectional=bidirectional, activation=activation)

model_file = sorted([p for p in os.listdir('results') if p.endswith('.h5')])[-1]
print('Using model file', model_file)
model.load_weights(os.path.join('results', model_file))

start = time.time()
pred = model.predict(data['X_test'], verbose=0)
end = time.time()
print('Predict done, shape', pred.shape, 'took', end-start, 's')
