from stock_prediction import create_model, load_data

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import pandas as pd
from parameters import *
from numba import cuda
from stockstats import StockDataFrame

from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def train(step, model_name):

    if os.path.isfile(ticker_data_filename):
        ticker = pd.read_csv(ticker_data_filename)
    data = load_data('^GDAXI', N_STEPS, n_days=step, test_size=TEST_SIZE, feature_columns=COLUMN_NAME)
   
    if not os.path.isfile(ticker_data_filename):
    # save the CSV file (dataset)
        data["dataframe"].to_csv(ticker_data_filename)

def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[N_STEPS]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model  
   
model = build_model()
model.summary()
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])