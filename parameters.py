'''
Created on Feb 14, 2020

@author: Yani Stratev
'''

import os
import time
from tensorflow.keras.layers import LSTM, GRU,RNN,Bidirectional
from keras import backend as K
import tensorflow as tf
# TAMAÑO DE LA VENTANA O SECUENCIA
N_STEPS = 70
#  SIGUIENTE DIA
N_DAYS_STEP= 2

# Usamos estas columnas 
COLUMN_NAME = ["adjclose", "volume", "open", "high", "low","macd","atr","dma"]
# tamaño de la ventana de testeo
TEST_SIZE = 0.2

# date now
date_now = time.strftime("%Y-%m-%d")
date_model="2020-03-15"
bidirectional = True
### model parameters
NUM_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
normalizer = 'adam'
### training parameters

# nombre de  lo que quiero sacar
ticker = "^GDAXI"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# mean squared error loss
LOSS = "mse"
OPTIMIZER = "sgd"
BATCH_SIZE = 64
EPOCHS = 350




