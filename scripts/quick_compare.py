"""Quick comparison between LSTM and GRU and between price vs returns targets.

Usage:
  python scripts/quick_compare.py [epochs]

This script trains small models (few epochs) on BTC-USD (or provider symbol) for 1-day horizon
and prints MAE (price-space) for each configuration.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(repo_root))
import os
import numpy as np
import importlib
import parameters
from stock_prediction import create_model, load_data
from tensorflow.keras.callbacks import ModelCheckpoint

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
PROVIDER = 'BTC-USD'

configs = [
    {'cell_name': 'LSTM', 'cell': parameters.CELL},
    {'cell_name': 'GRU', 'cell': __import__('tensorflow.keras.layers', fromlist=['GRU']).GRU}
]

targets = ['price', 'returns']

results = []
for cfg in configs:
    for tgt in targets:
        print(f"Running config: cell={cfg['cell_name']} target={tgt} epochs={EPOCHS}")
        try:
            data = load_data(PROVIDER, n_steps=parameters.N_STEPS, n_days=1, test_size=parameters.TEST_SIZE, feature_columns=parameters.COLUMN_NAME, shuffle=True, target=tgt)
        except Exception as e:
            print('Failed to load data for', PROVIDER, 'falling back to synthetic:', e)
            # build a synthetic DataFrame similar to other fallbacks
            import pandas as pd
            import numpy as np
            N = max(300, parameters.N_STEPS * 3)
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
            data = load_data(df, n_steps=parameters.N_STEPS, n_days=1, test_size=parameters.TEST_SIZE, feature_columns=parameters.COLUMN_NAME, shuffle=True, target=tgt)
        model = create_model(parameters.N_STEPS, loss=parameters.LOSS, units=parameters.UNITS, cell=cfg['cell'], num_layers=parameters.NUM_LAYERS, dropout=parameters.DROPOUT, normalizer=parameters.normalizer, bidirectional=parameters.bidirectional, activation=parameters.activation)
        # short train
        model.fit(np.array(data['X_train']), np.array(data['y_train']), epochs=EPOCHS, batch_size=parameters.BATCH_SIZE, verbose=0)
        # evaluate
        y_pred = model.predict(np.array(data['X_test']))
        if tgt == 'returns':
            y_pred_returns = data['y_scaler'].inverse_transform(y_pred).flatten()
            y_test_returns = data['y_scaler'].inverse_transform(np.array(data['y_test']).reshape(-1,1)).flatten()
            anchors = np.array(data['anchor_test'])
            y_pred_prices = anchors * (1 + y_pred_returns)
            y_test_prices = anchors * (1 + y_test_returns)
            mae = np.mean(np.abs(y_pred_prices - y_test_prices))
        else:
            y_pred_prices = parameters.COLUMN_NAME and data['column_scaler']['adjclose'].inverse_transform(y_pred)
            y_test_prices = data['column_scaler']['adjclose'].inverse_transform(np.array(data['y_test']).reshape(1,-1)).flatten()
            y_pred_prices = np.squeeze(data['column_scaler']['adjclose'].inverse_transform(y_pred))
            mae = np.mean(np.abs(y_pred_prices - y_test_prices))
        print(f"Config: {cfg['cell_name']} target={tgt} -> MAE (price): {mae}")
        sys.stdout.flush()
        results.append((cfg['cell_name'], tgt, mae))

print('\nSummary:')
for r in results:
    print(r)
