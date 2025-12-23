import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from stock_prediction import create_model
from parameters import N_STEPS, LOSS, UNITS, CELL, NUM_LAYERS, DROPOUT, normalizer, bidirectional, activation, ticker
import os

model_name = f"{ '2025-12-15' }_{ticker}-{LOSS}-{activation}-{normalizer}-{CELL.__name__}-seq-{N_STEPS}-step-1-layers-{NUM_LAYERS}-units-{UNITS}"
if bidirectional:
    model_name += 'bidirectional'
model_path = os.path.join('results', model_name) + '.h5'

print('Constructing model...')
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, num_layers=NUM_LAYERS,
                     dropout=DROPOUT, normalizer=normalizer, bidirectional=bidirectional, activation=activation)
print('Calling load_weights on', model_path)
model.load_weights(model_path)
print('Loaded weights successfully')
