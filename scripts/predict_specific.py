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

# Find the synthetic-fallback file
files = [f for f in os.listdir('results') if 'synthetic-fallback' in f and f.endswith('.h5')]
if not files:
    raise SystemExit('No synthetic-fallback model found in results/')
model_file = files[0]
print('Using model file', model_file)
model.load_weights(os.path.join('results', model_file))

start = time.time()
# Try predicting one sample to see per-sample time (avoid large compile overheads)
for i in range(min(3, len(data['X_test']))):
    s = time.time()
    p = model.predict(data['X_test'][i:i+1], verbose=0)
    e = time.time()
    print(f'predict sample {i} done shape {p.shape} time {(e-s):.4f}s')

end = time.time()
print('Per-sample predictions done, total time', end-start)
