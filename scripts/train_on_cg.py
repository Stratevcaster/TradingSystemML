"""Train a BTC model using CoinGecko data (last 365 days) and save model to results/"""
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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TARGET = sys.argv[2] if len(sys.argv) > 2 else 'returns'
CELL_NAME = sys.argv[3] if len(sys.argv) > 3 else None
if CELL_NAME and CELL_NAME.upper() == 'GRU':
    parameters.CELL = __import__('tensorflow.keras.layers', fromlist=['GRU']).GRU

print('Fetching CoinGecko BTC data...')
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

print('Building dataset (target=', TARGET, ')')
parameters.TARGET = TARGET
data = load_data(df, n_steps=parameters.N_STEPS, n_days=1, test_size=parameters.TEST_SIZE, feature_columns=parameters.COLUMN_NAME, shuffle=False, target=TARGET)
print('Shapes: X_train', np.array(data['X_train']).shape, 'y_train', np.array(data['y_train']).shape)

model = create_model(parameters.N_STEPS, loss=parameters.LOSS if TARGET=='price' else 'huber', units=parameters.UNITS, cell=parameters.CELL, num_layers=parameters.NUM_LAYERS, dropout=parameters.DROPOUT, normalizer=parameters.normalizer, bidirectional=parameters.bidirectional, activation=parameters.activation)
model_name = f"{parameters.date_now}_BTC-{parameters.LOSS}-{parameters.activation}-{parameters.normalizer}-{parameters.CELL.__name__}-seq-{parameters.N_STEPS}-step-1-layers-{parameters.NUM_LAYERS}-units-{parameters.UNITS}"
if parameters.bidirectional:
    model_name += 'bidirectional'
checkpointer = ModelCheckpoint(os.path.join('results', model_name)+'.keras', save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
print('Training model...')
model.fit(np.array(data['X_train']), np.array(data['y_train']), epochs=EPOCHS, batch_size=parameters.BATCH_SIZE, validation_data=(np.array(data['X_test']), np.array(data['y_test'])), callbacks=[checkpointer, early, reduce_lr], verbose=2)
save_path = os.path.join('results', model_name) + '.h5'
model.save(save_path)
print('Saved model to', save_path)
