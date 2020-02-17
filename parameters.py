'''
Created on Feb 14, 2020

@author: USER
'''

import os
import time
from tensorflow.keras.layers import LSTM
from keras import backend as K
import tensorflow as tf
# Window size or the sequence length
N_STEPS = 20
# Lookup step, 1 is the next day
LOOKUP_STEP = 10

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4

### training parameters

# mean squared error loss
LOSS = "mse"
OPTIMIZER = "rmsprop"
BATCH_SIZE = 64
EPOCHS = 30

# Apple stock market
ticker = "TSLA"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")


