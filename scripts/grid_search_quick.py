"""Quick grid search over a small hyperparameter space using CoinGecko BTC 365d data.
Saves results to results/grid_search_{date}.csv
"""
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    target = sys.argv[2] if len(sys.argv) > 2 else 'returns'
    # small grid
    units_list = [64, 128, 256]
    layers_list = [1, 2]
    dropout_list = [0.1, 0.2]
    cell_types = ['LSTM', 'GRU']

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

    for cell_name in cell_types:
        for units in units_list:
            for num_layers in layers_list:
                for dropout in dropout_list:
                    print(f'Running {cell_name} units={units} layers={num_layers} dropout={dropout}')
                    if cell_name == 'GRU':
                        cell = __import__('tensorflow.keras.layers', fromlist=['GRU']).GRU
                    else:
                        cell = __import__('tensorflow.keras.layers', fromlist=['LSTM']).LSTM

                    loss = 'huber' if target == 'returns' else parameters.LOSS
                    model = create_model(parameters.N_STEPS, units=units, cell=cell, num_layers=num_layers, dropout=dropout, loss=loss, normalizer=parameters.normalizer, bidirectional=parameters.bidirectional, activation=parameters.activation)

                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0)
                    ]

                    history = model.fit(X_train, y_train, epochs=epochs, batch_size=parameters.BATCH_SIZE, validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)

                    # raw evaluation
                    eval_res = model.evaluate(X_test, y_test, verbose=0)
                    mae_scaled = float(eval_res[1]) if len(eval_res) > 1 else None

                    # compute price-space MAE if target is returns
                    if target == 'returns' and y_scaler is not None:
                        # invert scaled returns
                        preds_scaled = model.predict(X_test)
                        preds_unscaled = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
                        y_unscaled = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
                        preds_price = anchor_test * (1 + preds_unscaled)
                        true_price = anchor_test * (1 + y_unscaled)
                        mae_price = float(np.mean(np.abs(preds_price - true_price)))
                        rmse_price = float(np.sqrt(np.mean((preds_price - true_price)**2)))
                    else:
                        # price target: y_test are scaled prices (MinMax); we can unscale using column scaler
                        mae_price = None
                        rmse_price = None

                    results.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'cell': cell_name,
                        'units': units,
                        'layers': num_layers,
                        'dropout': dropout,
                        'mae_scaled': mae_scaled,
                        'mae_price': mae_price,
                        'rmse_price': rmse_price
                    })

    df_out = pd.DataFrame(results)
    out_path = os.path.join('results', f'grid_search_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv')
    df_out.to_csv(out_path, index=False)
    print('Saved grid search results to', out_path)


if __name__ == '__main__':
    main()
