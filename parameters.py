'''
Created on Feb 14, 2020

@author: Yani Stratev
'''

import os
import time
from tensorflow.keras.layers import LSTM, GRU,RNN,Bidirectional
from keras import backend as K
import tensorflow as tf
# TAMA�O DE LA VENTANA O SECUENCIA
N_STEPS = 70
#  SIGUIENTE DIA
N_DAYS_STEP= 18

# Usamos estas columnas 
COLUMN_NAME = ["adjclose", "volume", "open", "high", "low","macd","atr","dma","sma10","sma30","rsi14"]
# tama�o de la ventana de testeo
TEST_SIZE = 0.2

# date now
date_now = time.strftime("%Y-%m-%d")
date_model="2020-04-12"
bidirectional = True
### model parameters
NUM_LAYERS = 2
# Default recurrent cell (can be set to GRU for experiments)
CELL =  LSTM
# 128 RNN neurons (smaller default for faster, more stable runs)
UNITS = 128
# 20% dropout (reduced from 40% to reduce underfitting risk)
DROPOUT = 0.2
normalizer = 'adam'
## model parameters
### training parameters

# nombre de  lo que quiero sacar
ticker = "BTC"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# mean squared error loss (can try 'mae' or 'huber' for robustness)
LOSS = "mse"
# OPTIMIZER = "sgd"
BATCH_SIZE = 64
# Default epochs (use small values for quick experiments; scripts override this)
EPOCHS = 100

# Model output activation: use 'linear' for regression targets (was 'relu', which can clip outputs)
activation = 'linear'

# Target type: 'price' (predict future price) or 'returns' (predict pct-change)
# Use 'price' by default to preserve existing behavior; we'll experiment with 'returns'
TARGET = 'price'




