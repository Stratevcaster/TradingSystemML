"""Quick smoke-test trainer: temporarily reduces model size and epochs to perform a fast check.

This script mutates values in `parameters` at runtime so it doesn't change the repository state.
Run inside the `trading` conda env created by the setup script.

Usage:
  python scripts/run_quick_train.py
"""
import sys
from pathlib import Path

# Ensure repository root is on sys.path so imports like `train` and `parameters` work
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from train import train
import parameters
import os
import pandas as pd
import numpy as np
from stock_prediction import create_model, load_data

# Make directories expected by the training script
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Quick-test overrides (do not modify files)
parameters.EPOCHS = 1
parameters.UNITS = 8
parameters.BATCH_SIZE = 16
parameters.NUM_LAYERS = 1

model_name = "quicktest"

print("Running quick train (1 epoch, small model) to smoke-test training pipeline (using synthetic data)...")
try:
    # Create synthetic dataset to avoid network dependencies for quick smoke tests
    N = max(300, parameters.N_STEPS * 3)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=N)
    df = pd.DataFrame(index=dates)
    np.random.seed(42)
    df['adjclose'] = np.cumsum(np.random.normal(0, 1, size=N)) + 100
    df['volume'] = np.random.randint(1000, 10000, size=N)
    df['open'] = df['adjclose'] + np.random.normal(0, 1, size=N)
    df['high'] = df[['open', 'adjclose']].max(axis=1) + np.random.rand(N)
    df['low'] = df[['open', 'adjclose']].min(axis=1) - np.random.rand(N)
    # Ensure additional feature columns exist
    for col in ['macd', 'atr', 'dma']:
        df[col] = 0.0

    # Use load_data with a DataFrame to create train/test tensors
    data = load_data(df, n_steps=parameters.N_STEPS, shuffle=False, n_days=1, test_size=0.2, feature_columns=parameters.COLUMN_NAME)

    # Build a small model and train for 1 epoch (overrides already set in parameters)
    model = create_model(parameters.N_STEPS, loss=parameters.LOSS, units=parameters.UNITS, cell=parameters.CELL, num_layers=parameters.NUM_LAYERS, dropout=parameters.DROPOUT, normalizer=parameters.normalizer if hasattr(parameters,'normalizer') else parameters.normalizer, bidirectional=parameters.bidirectional, activation=parameters.activation)

    model.fit(data['X_train'], data['y_train'], batch_size=parameters.BATCH_SIZE, epochs=parameters.EPOCHS, validation_data=(data['X_test'], data['y_test']), verbose=1)
    model.save(os.path.join("results", model_name) + ".h5")
    print("Quick train finished. Check ./results for the model files.")
except Exception as e:
    print("Quick train failed with exception:")
    import traceback
    traceback.print_exc()
    raise
