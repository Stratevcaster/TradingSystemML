"""Retrain the chosen top configuration and save evaluation artifacts.

Defaults:
- cell: LSTM
- units: 128
- layers: 1
- dropout: 0.1
- target: returns
- epochs: 30

Saves model, metrics, predictions CSV and prediction plot to `results/`.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(repo_root))

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
import parameters
from stock_prediction import create_model, load_data


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
    # ensure compatibility with StockDataFrame which expects a 'close' column
    df['close'] = df['adjclose']
    df['open'] = df['adjclose']
    df['high'] = df['adjclose']
    df['low'] = df['adjclose']
    if vols:
        vol_df = pd.DataFrame(vols, columns=['date_ms','volume'])
        vol_df['date'] = pd.to_datetime(vol_df['date_ms'], unit='ms')
        vol_df.set_index('date', inplace=True)
        df['volume'] = vol_df['volume']
    else:
        df['volume'] = 0
    return df


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    target = sys.argv[2] if len(sys.argv) > 2 else 'returns'
    units = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    layers = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    dropout = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1

    print(f'Retraining top config: cell=LSTM units={units} layers={layers} dropout={dropout} target={target} epochs={epochs}')
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # fetch data
    df = fetch_cg_last365()

    # set parameters for this run
    parameters.TARGET = target
    parameters.UNITS = units
    parameters.NUM_LAYERS = layers
    parameters.DROPOUT = dropout
    parameters.CELL = __import__('tensorflow.keras.layers', fromlist=['LSTM']).LSTM

    data = load_data(df, n_steps=parameters.N_STEPS, n_days=1, test_size=parameters.TEST_SIZE, feature_columns=parameters.COLUMN_NAME, shuffle=False, target=target)

    X_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])
    X_test = np.array(data['X_test'])
    y_test = np.array(data['y_test'])
    anchor_test = np.array(data['anchor_test'])
    y_scaler = data.get('y_scaler')

    loss = 'huber' if target == 'returns' else parameters.LOSS
    model = create_model(parameters.N_STEPS, units=units, cell=parameters.CELL, num_layers=layers, dropout=dropout, loss=loss, normalizer=parameters.normalizer, bidirectional=parameters.bidirectional, activation=parameters.activation)

    model_name = f"{parameters.date_now}_BTC-retrain-top-{parameters.CELL.__name__}-units-{units}-layers-{layers}-dropout-{dropout}"
    save_best = os.path.join('results', model_name) + '.keras'
    checkpointer_cb = __import__('tensorflow.keras.callbacks', fromlist=['ModelCheckpoint']).ModelCheckpoint(save_best, save_best_only=True, verbose=1)
    early = __import__('tensorflow.keras.callbacks', fromlist=['EarlyStopping']).EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    print('Starting training...')
    model.fit(X_train, y_train, epochs=epochs, batch_size=parameters.BATCH_SIZE, validation_data=(X_test, y_test), callbacks=[checkpointer_cb, early], verbose=2)

    # final save
    final_path = os.path.join('results', model_name) + '.h5'
    model.save(final_path)

    # evaluate and create artifacts
    print('Evaluating on test set...')
    preds_scaled = model.predict(X_test)
    preds_scaled = np.asarray(preds_scaled).reshape(-1)
    y_test_arr = np.asarray(y_test).reshape(-1)
    anchor_arr = np.asarray(anchor_test).reshape(-1)
    min_len = min(len(preds_scaled), len(y_test_arr), len(anchor_arr))
    preds_scaled = preds_scaled[:min_len]
    y_test_arr = y_test_arr[:min_len]
    anchor_arr = anchor_arr[:min_len]

    if target == 'returns' and y_scaler is not None:
        preds_unscaled = y_scaler.inverse_transform(preds_scaled.reshape(-1,1)).reshape(-1)
        y_unscaled = y_scaler.inverse_transform(y_test_arr.reshape(-1,1)).reshape(-1)
        preds_price = anchor_arr * (1 + preds_unscaled)
        true_price = anchor_arr * (1 + y_unscaled)
    else:
        # price target (if used) - y_test is scaled minmax for adjclose; attempt to unscale using the column scaler
        preds_price = preds_scaled
        true_price = y_test_arr

    mae_price = float(np.mean(np.abs(preds_price - true_price)))
    rmse_price = float(np.sqrt(np.mean((preds_price - true_price)**2)))
    # directional accuracy
    dir_acc = float(np.mean(np.sign(preds_price[1:] - preds_price[:-1]) == np.sign(true_price[1:] - true_price[:-1])))

    metrics = {
        'model_name': model_name,
        'mae_price': mae_price,
        'rmse_price': rmse_price,
        'directional_accuracy': dir_acc,
        'timestamp': datetime.utcnow().isoformat()
    }

    metrics_path = os.path.join('results', model_name + '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # save predictions CSV
    preds_df = pd.DataFrame({'anchor': anchor_arr, 'true_price': true_price, 'pred_price': preds_price})
    preds_csv = os.path.join('results', model_name + '_preds.csv')
    preds_df.to_csv(preds_csv, index=False)

    # plot
    plt.figure(figsize=(10,6))
    plt.plot(true_price, label='true')
    plt.plot(preds_price, label='pred')
    plt.legend()
    plt.title(f'Predictions ({model_name})')
    plot_path = os.path.join('results','plots', model_name + '_preds.png')
    plt.savefig(plot_path)
    plt.close()

    print('Retrain complete. Artifacts saved:')
    print(' - model:', final_path)
    print(' - metrics:', metrics_path)
    print(' - preds CSV:', preds_csv)
    print(' - plot:', plot_path)


if __name__ == '__main__':
    main()
