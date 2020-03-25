'''
Created on Feb 14, 2020
https://www.datacamp.com/community/tutorials/lstm-python-stock-market
MIRAR
@author: USER
'''
from stock_prediction import create_model, load_data
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import pandas as pd
from parameters import *
from numba import cuda
from stockstats import StockDataFrame


def train(step, model_name):

    if os.path.isfile(ticker_data_filename):
        ticker = pd.read_csv(ticker_data_filename)
    data = load_data('^GDAXI', N_STEPS, n_days=step, test_size=TEST_SIZE, feature_columns=COLUMN_NAME)
   
    if not os.path.isfile(ticker_data_filename):
    # save the CSV file (dataset)
        data["dataframe"].to_csv(ticker_data_filename)

    #model_name = f"{date_now}_{ticker}-{LOSS}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
 
    model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, num_layers=NUM_LAYERS,
                dropout=DROPOUT,normalizer=normalizer,bidirectional=bidirectional)

    # some tensorflow callbacks
    #model_name = f"{date_now}_{ticker}-{LOSS}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"

    checkpointer = ModelCheckpoint(os.path.join("results", model_name), save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

    history = model.fit(data["X_train"], data["y_train"],
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(data["X_test"], data["y_test"]),
                callbacks=[checkpointer, tensorboard],
                verbose=1)

    model.save(os.path.join("results", model_name) + ".h5")
    K.clear_session()
    cuda.select_device(0)
    #  cuda.close()
