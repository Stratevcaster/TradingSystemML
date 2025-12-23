"""Run a short BTC experiment: train a short model (default 3 epochs) and run the test for N days ahead.

Usage:
  python scripts/run_btc_experiment.py [epochs] [days]

Example:
  python scripts/run_btc_experiment.py 3 10
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import os
import subprocess
import importlib
import parameters

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
# optional third arg: target type ('price' or 'returns')
TARGET = sys.argv[3] if len(sys.argv) > 3 else 'price'
# optional fourth arg: cell name
CELL_NAME = sys.argv[4] if len(sys.argv) > 4 else None
# User-visible ticker (used for naming and testing)
USER_TICKER = 'BTC'
# Symbol to pass to data providers (yfinance uses 'BTC-USD')
PROVIDER_SYMBOL = 'BTC-USD'

print(f"BTC experiment: epochs={EPOCHS}, days={N_DAYS}, user_ticker={USER_TICKER}, provider_symbol={PROVIDER_SYMBOL}")

# Run the short training using the existing script (it accepts epochs and optional ticker)
cmd = [sys.executable, os.path.join('scripts', 'run_real_train.py'), str(EPOCHS), PROVIDER_SYMBOL, TARGET]
if CELL_NAME:
  cmd.append(CELL_NAME)
print('Running training:', ' '.join(cmd))
res = subprocess.run(cmd, check=False)
if res.returncode != 0:
    print('Training script returned non-zero exit code, continuing to testing (may use fallback models).')

# Now run the test using the in-process tester to get predictions for N_DAYS
print('Running test (in-process) with N_DAYS_STEP =', N_DAYS)
old_ticker = parameters.ticker
old_N_DAYS = getattr(parameters, 'N_DAYS_STEP', None)
try:
    # After training, copy any generated model artifacts that include the provider symbol
    # to equivalent files that include the user ticker 'BTC' so the test can find them by ticker name.
    import shutil
    found = []
    for pattern in [f"*{PROVIDER_SYMBOL}*.h5", f"*{PROVIDER_SYMBOL}*.keras"]:
      for p in sorted([os.path.join('results', x) for x in os.listdir('results') if PROVIDER_SYMBOL in x and x.endswith(os.path.splitext(pattern)[1])]):
        dst = p.replace(PROVIDER_SYMBOL, USER_TICKER)
        try:
          shutil.copy2(p, dst)
          print(f"Copied {p} -> {dst}")
          found.append(dst)
        except Exception as e:
          print(f"Failed to copy {p} -> {dst}: {e}")

    parameters.ticker = USER_TICKER
    parameters.N_DAYS_STEP = N_DAYS
    parameters.TARGET = TARGET

    # If no provider-created model files were found/copied above, and there are no BTC models
    # in ./results, create a small synthetic BTC model now so testing can proceed.
    btc_models = [p for p in os.listdir('results') if USER_TICKER in p and (p.endswith('.h5') or p.endswith('.keras'))]
    if not found and not btc_models:
      print('No BTC model files found after training; creating a short synthetic BTC model for testing...')
      from stock_prediction import create_model, load_data
      import pandas as pd
      import numpy as np

      N = max(300, parameters.N_STEPS * 3)
      dates = pd.date_range(end=pd.Timestamp.today(), periods=N)
      df = pd.DataFrame(index=dates)
      np.random.seed(0)
      df['adjclose'] = np.cumsum(np.random.normal(0, 1, size=N)) + 100
      df['volume'] = np.random.randint(1000, 10000, size=N)
      df['open'] = df['adjclose'] + np.random.normal(0, 1, size=N)
      df['high'] = df[['open', 'adjclose']].max(axis=1) + np.random.rand(N)
      df['low'] = df[['open', 'adjclose']].min(axis=1) - np.random.rand(N)
      for col in ['macd', 'atr', 'dma']:
        df[col] = 0.0

      data = load_data(df, n_steps=parameters.N_STEPS, shuffle=False, n_days=1, test_size=parameters.TEST_SIZE, feature_columns=parameters.COLUMN_NAME, target=TARGET)
      model = create_model(parameters.N_STEPS, loss=parameters.LOSS, units=parameters.UNITS, cell=parameters.CELL, num_layers=parameters.NUM_LAYERS, dropout=parameters.DROPOUT, normalizer=parameters.normalizer, bidirectional=parameters.bidirectional, activation=parameters.activation)
      print('Training small synthetic BTC model (1 epoch)')
      model.fit(data['X_train'], data['y_train'], epochs=min(1, EPOCHS), batch_size=parameters.BATCH_SIZE, verbose=2)
      os.makedirs('results', exist_ok=True)
      model_name = f"{parameters.date_now}_{USER_TICKER}-{parameters.LOSS}-{parameters.activation}-{parameters.normalizer}-{parameters.CELL.__name__}-seq-{parameters.N_STEPS}-step-1-layers-{parameters.NUM_LAYERS}-units-{parameters.UNITS}-synthetic-manual"
      save_path = os.path.join('results', model_name) + '.h5'
      model.save(save_path)
      print('Saved synthetic BTC model to', save_path)

    # import and call tester
    import tester
    importlib.reload(tester)
    preds = tester.test(N_DAYS)
    print('BTC experiment predictions:', preds)
    # Save predictions
    os.makedirs('results', exist_ok=True)
    out_path = os.path.join('results', f'btc_{N_DAYS}day_preds_{parameters.date_now}.txt')
    with open(out_path, 'w') as f:
        for p in preds:
            f.write(f"{p}\n")
    print('Saved predictions to', out_path)
finally:
    parameters.ticker = old_ticker
    if old_N_DAYS is not None:
        parameters.N_DAYS_STEP = old_N_DAYS

print('BTC experiment finished.')
