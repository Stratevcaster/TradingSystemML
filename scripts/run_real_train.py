"""Run a short real training job using market data to verify end-to-end behavior.

This script overrides `parameters.EPOCHS` to a small number (3 by default) so it's quick
enough for verification. It saves the model into `results/` using the same naming format
as the orchestrator.

Usage:
  python scripts/run_real_train.py [epochs]

Example:
  python scripts/run_real_train.py 3
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import os
from parameters import date_now, LOSS, CELL, N_STEPS, NUM_LAYERS, UNITS, ticker, N_DAYS_STEP, COLUMN_NAME, bidirectional, normalizer, activation
import parameters
from stock_prediction import create_model, load_data
import yahoo_fin.stock_info as si
import yfinance as yf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import pandas as pd
from parameters import *
from keras import backend as K
from numba import cuda
import numpy as np

def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    # override parameters for short run
    parameters.EPOCHS = epochs

    # ensure folders
    for d in ("logs", "results", "data"):
        os.makedirs(d, exist_ok=True)

    step = 1
    ticker_to_use = sys.argv[2] if len(sys.argv) > 2 else ticker
    # optional third arg: target type ('price' or 'returns')
    target_to_use = sys.argv[3] if len(sys.argv) > 3 else getattr(parameters, 'TARGET', 'price')
    # optional fourth arg: cell name ('LSTM' or 'GRU')
    cell_name = sys.argv[4] if len(sys.argv) > 4 else None
    if cell_name is not None:
        if cell_name.upper() == 'GRU':
            parameters.CELL = __import__('tensorflow.keras.layers', fromlist=['GRU']).GRU
        elif cell_name.upper() == 'LSTM':
            parameters.CELL = __import__('tensorflow.keras.layers', fromlist=['LSTM']).LSTM
    model_name = f"{date_now}_{ticker}-{LOSS}-{activation}-{normalizer}-{CELL.__name__}-seq-{N_STEPS}-step-{step}-layers-{NUM_LAYERS}-units-{UNITS}"
    if bidirectional:
        model_name += 'bidirectional'

    parameters.TARGET = target_to_use
    print(f"Starting short real training: epochs={epochs}, step={step}, model_name={model_name}, target={parameters.TARGET}, cell={parameters.CELL.__name__}")
    sys.stdout.flush()

    # Get market data. For crypto-like tickers, try CoinGecko first because yfinance/yahoo_fin
    # sometimes fail to return crypto history.
    df = None
    if 'BTC' in ticker_to_use.upper() or 'ETH' in ticker_to_use.upper():
        try:
            print(f"Ticker {ticker_to_use} looks like crypto — trying CoinGecko first...")
            from pycoingecko import CoinGeckoAPI
            cg = CoinGeckoAPI()
            chart = cg.get_coin_market_chart_by_id('bitcoin', vs_currency='usd', days='max')
            prices = chart.get('prices', [])
            vols = chart.get('total_volumes', [])
            if prices:
                df = pd.DataFrame(prices, columns=['date_ms', 'adjclose'])
                df['date'] = pd.to_datetime(df['date_ms'], unit='ms')
                df.set_index('date', inplace=True)
                df = df[['adjclose']]
                df['open'] = df['adjclose']
                df['high'] = df['adjclose']
                df['low'] = df['adjclose']
                if vols:
                    vol_df = pd.DataFrame(vols, columns=['date_ms', 'volume'])
                    vol_df['date'] = pd.to_datetime(vol_df['date_ms'], unit='ms')
                    vol_df.set_index('date', inplace=True)
                    df = df.join(vol_df['volume'].rename('volume'), how='left')
                else:
                    df['volume'] = 0
                df = df.reset_index()
                print(f"CoinGecko returned {len(df)} rows for {ticker_to_use}")
        except Exception as e:
            print('CoinGecko primary attempt failed:', repr(e))

    if df is None:
        try:
            print(f"Fetching data for {ticker_to_use} via yahoo_fin...")
            df = si.get_data(ticker_to_use)
        except Exception as e:
            print("yahoo_fin failed, falling back to yfinance: ", repr(e))
            try:
                df = yf.download(ticker_to_use, period='5y', interval='1d')
                # normalize columns to the format expected by load_data
                if 'Adj Close' in df.columns:
                    df.rename(columns={'Adj Close': 'adjclose'}, inplace=True)
                if 'Close' in df.columns and 'adjclose' not in df.columns:
                    df.rename(columns={'Close': 'adjclose'}, inplace=True)
                if 'Volume' in df.columns:
                    df.rename(columns={'Volume': 'volume'}, inplace=True)
                df['open'] = df.get('Open', df['adjclose'])
                df['high'] = df.get('High', df['adjclose'])
                df['low'] = df.get('Low', df['adjclose'])
                df = df.reset_index()
                # if yfinance returned an empty dataframe, try CoinGecko as a fallback
                if df is None or len(df) == 0 or 'adjclose' not in df.columns:
                    print('yfinance returned empty data; attempting CoinGecko fallback (bitcoin -> USD)')
                    try:
                        from pycoingecko import CoinGeckoAPI
                        cg = CoinGeckoAPI()
                        chart = cg.get_coin_market_chart_by_id('bitcoin', vs_currency='usd', days='max')
                        prices = chart.get('prices', [])
                        vols = chart.get('total_volumes', [])
                        if not prices:
                            raise RuntimeError('CoinGecko returned no price data')
                        df = pd.DataFrame(prices, columns=['date_ms', 'adjclose'])
                        df['date'] = pd.to_datetime(df['date_ms'], unit='ms')
                        df.set_index('date', inplace=True)
                        df = df[['adjclose']]
                        df['open'] = df['adjclose']
                        df['high'] = df['adjclose']
                        df['low'] = df['adjclose']
                        if vols:
                            vol_df = pd.DataFrame(vols, columns=['date_ms', 'volume'])
                            vol_df['date'] = pd.to_datetime(vol_df['date_ms'], unit='ms')
                            vol_df.set_index('date', inplace=True)
                            df['volume'] = vol_df['volume']
                        else:
                            df['volume'] = 0
                        df = df.reset_index()
                        print(f"CoinGecko fallback returned {len(df)} rows")
                    except Exception as e3:
                        print('CoinGecko fallback failed after empty yfinance result:', repr(e3))
            except Exception as e2:
                print("yfinance fallback also failed:", repr(e2))
                # Try CoinGecko as a last-resort fallback (works for crypto like BTC)
                try:
                    print('Attempting CoinGecko fallback (bitcoin -> USD)')
                    from pycoingecko import CoinGeckoAPI
                    cg = CoinGeckoAPI()
                    chart = cg.get_coin_market_chart_by_id('bitcoin', vs_currency='usd', days='max')
                    prices = chart.get('prices', [])
                    vols = chart.get('total_volumes', [])
                    if not prices:
                        raise RuntimeError('CoinGecko returned no price data')
                    df = pd.DataFrame(prices, columns=['date_ms', 'adjclose'])
                    df['date'] = pd.to_datetime(df['date_ms'], unit='ms')
                    df.set_index('date', inplace=True)
                    df = df[['adjclose']]
                    df['open'] = df['adjclose']
                    df['high'] = df['adjclose']
                    df['low'] = df['adjclose']
                    if vols:
                        vol_df = pd.DataFrame(vols, columns=['date_ms', 'volume'])
                        vol_df['date'] = pd.to_datetime(vol_df['date_ms'], unit='ms')
                        vol_df.set_index('date', inplace=True)
                        df['volume'] = vol_df['volume']
                    else:
                        df['volume'] = 0
                    df = df.reset_index()
                except Exception as e3:
                    print('CoinGecko fallback also failed:', repr(e3))
                    raise

    # Validate fetched data; if empty or too small, fallback to synthetic dataset
    def _is_valid_df(df):
        try:
            return df is not None and len(df) >= N_STEPS + 5
        except Exception:
            return False

    if not _is_valid_df(df):
        print("Fetched data is empty or too small; falling back to synthetic data for the short run.")
        sys.stdout.flush()
        # Build synthetic dataset
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
        model_name += '-synthetic-fallback'

    # Build dataset using load_data (pass DataFrame to avoid re-fetching)
    try:
        data = load_data(df, n_steps=N_STEPS, shuffle=False, n_days=step, test_size=TEST_SIZE, feature_columns=COLUMN_NAME, target=getattr(parameters, 'TARGET', 'price'))
    except Exception:
        print("Failed to build dataset from fetched data; aborting.")
        raise

    # choose loss: for returns it's often better to use Huber for robustness
    local_loss = LOSS
    if parameters.TARGET == 'returns' and LOSS == 'mse':
        local_loss = 'huber'

    # Build the model
    model = create_model(N_STEPS, loss=local_loss, units=UNITS, cell=CELL, num_layers=NUM_LAYERS,
                         dropout=DROPOUT, normalizer=normalizer, bidirectional=bidirectional, activation=activation)

    # callbacks
    checkpointer = ModelCheckpoint(os.path.join("results", model_name) + ".keras", save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

    # Train (with EarlyStopping to avoid overfitting and speed up experiments)
    try:
        print(f"X_train shape: {getattr(data['X_train'], 'shape', None)}, y_train shape: {getattr(data['y_train'], 'shape', None)}")
        print(f"X_test shape: {getattr(data['X_test'], 'shape', None)}, y_test shape: {getattr(data['y_test'], 'shape', None)}")
        sys.stdout.flush()
        early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        hist = model.fit(data["X_train"], data["y_train"],
              batch_size=BATCH_SIZE,
              epochs=parameters.EPOCHS,
              validation_data=(data["X_test"], data["y_test"]),
              callbacks=[checkpointer, tensorboard, early],
              verbose=2)
        save_path = os.path.join("results", model_name) + ".h5"
        model.save(save_path)
        print(f"Model saved to {save_path}")
        K.clear_session()
        try:
            cuda.select_device(0)
        except Exception:
            pass
        print("Training finished. Check './results' for the saved model (and ./logs for TensorBoard).")
    except Exception:
        print("Training run raised an exception:")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
