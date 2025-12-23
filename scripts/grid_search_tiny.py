"""Very small and fast grid search for quick checks (LSTM only, short epochs)."""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(repo_root))
import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
import parameters
from stock_prediction import create_model, load_data
import os
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime


def fetch_cg_last365():
    cg = CoinGeckoAPI()
    chart = cg.get_coin_market_chart_by_id('bitcoin', vs_currency='usd', days='365')
    prices = chart.get('prices', [])
    vols = chart.get('total_volumes', [])
    if not prices:
        raise RuntimeError('CoinGecko returned no prices')
    df = pd.DataFrame(prices, columns=['date_ms','adjclose'])
    df['date'] = pd.to_datetime(df['date_ms'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[['adjclose']]
    df['open'] = df['adjclose']
    df['high'] = df['adjclose']
    df['low'] = df['adjclose']
    df['close'] = df['adjclose']
    if vols:
        vol_df = pd.DataFrame(vols, columns=['date_ms','volume'])
        vol_df['date'] = pd.to_datetime(vol_df['date_ms'], unit='ms')
        vol_df.set_index('date', inplace=True)
        df['volume'] = vol_df['volume']
    else:
        df['volume'] = 0
    return df


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    target = sys.argv[2] if len(sys.argv) > 2 else 'returns'

    print('Fetching data...')
    df = fetch_cg_last365()

    parameters.TARGET = target
    data = load_data(df, n_steps=parameters.N_STEPS, n_days=1, test_size=parameters.TEST_SIZE, feature_columns=parameters.COLUMN_NAME, shuffle=False, target=target)

    X_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])
    X_test = np.array(data['X_test'])
    y_test = np.array(data['y_test'])
    anchor_test = np.array(data['anchor_test'])
    y_scaler = data.get('y_scaler')

    results = []

    # tiny grid: LSTM only, 128 units, layers 1 and 2
    for num_layers in [1, 2]:
        units = 128
        dropout = 0.1
        print(f'Running LSTM units={units} layers={num_layers} dropout={dropout}')
        cell = __import__('tensorflow.keras.layers', fromlist=['LSTM']).LSTM

        loss = 'huber' if target == 'returns' else parameters.LOSS
        model = create_model(parameters.N_STEPS, units=units, cell=cell, num_layers=num_layers, dropout=dropout, loss=loss, normalizer=parameters.normalizer, bidirectional=parameters.bidirectional, activation=parameters.activation)

        callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)]

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=parameters.BATCH_SIZE, validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)

        preds_scaled = model.predict(X_test)
        # normalize shapes and avoid broadcasting errors by trimming to the smallest common length
        preds_scaled = np.asarray(preds_scaled).reshape(-1)
        y_test_arr = np.asarray(y_test).reshape(-1)
        anchor_arr = np.asarray(anchor_test).reshape(-1)
        min_len = min(len(preds_scaled), len(y_test_arr), len(anchor_arr))
        preds_scaled = preds_scaled[:min_len]
        y_test_arr = y_test_arr[:min_len]
        anchor_arr = anchor_arr[:min_len]
        preds_unscaled = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
        y_unscaled = y_scaler.inverse_transform(y_test_arr.reshape(-1, 1)).reshape(-1)
        preds_price = anchor_arr * (1 + preds_unscaled)
        true_price = anchor_arr * (1 + y_unscaled)
        mae_price = float(np.mean(np.abs(preds_price - true_price)))
        rmse_price = float(np.sqrt(np.mean((preds_price - true_price)**2)))

        results.append({
            'timestamp': datetime.utcnow().isoformat(),
            'cell': 'LSTM',
            'units': units,
            'layers': num_layers,
            'dropout': dropout,
            'mae_price': mae_price,
            'rmse_price': rmse_price
        })

    df_out = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    out_path = os.path.join('results', f'grid_search_tiny_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv')
    df_out.to_csv(out_path, index=False)
    print('Saved tiny grid search results to', out_path)


if __name__ == '__main__':
    main()
