'''
Created on Feb 14, 2020

@author: YANI STRATEV
ULTIMO FUNCIONAL 
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
import requests
from stockstats import StockDataFrame
import numpy as np
import pandas as pd
import random
import datetime
from parameters import bidirectional,activation
from pandas.tests.frame.test_validate import dataframe


#Funcion para Tingo a Json a Frame
# @return frame

def get_stock_dataJSON(stock_sym, start_date, end_date,index_as_date = True):
    base_url = 'https://api.tiingo.com/tiingo/daily/'+stock_sym + '/prices?'
    token = 'da2ac110cdd4a6586434808a9c2a275af4fc5693'
    payload = {
        'token' : token,
        'startDate' : start_date,
        'endDate' : end_date
        }
    response = requests.get(base_url,params=payload)
    data = response.json()
    #df = pd.DataFrame.from_records(data).T         
    df = pd.io.json.json_normalize(data)
    dates = []   
    ticker = stock_sym
    for arr_date in df['date']:
       
        final_date = arr_date.split("T")[0]
        final_date = datetime.datetime.strptime(final_date, "%Y-%m-%d")
        final_date = datetime.datetime.timestamp(final_date)
        dates.append(final_date)
    df["date"] = dates
    # get open / high / low / close data
    
    # get the date info
    temp_time = df['date']
    df.index = pd.to_datetime(temp_time, unit = "s")
    df.index = df.index.map(lambda dt: dt.floor("d"))
    
    
    frame = df[['close', 'volume', 'open', 'high', 'low']]
        
    frame['ticker'] = ticker.upper()
    
    if not index_as_date:  
        frame = frame.reset_index()
        frame.rename(columns = {"index": "date"}, inplace = True)
        
    return frame


def load_data(ticker, n_steps=70, shuffle=True, n_days=10, 
                test_size=0.3, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
  
    # Comprobar si se trata de un strig o se le pasa un DataFrame
    if isinstance(ticker, str):
        # cargar de la liberia
        dataframe = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        dataframe = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    dataframe = StockDataFrame.retype(dataframe)
    dataframe['macd'] = dataframe.get('macd') # calculate MACD
    dataframe['atr'] = dataframe.get('atr') # calculate ATR
    dataframe['dma'] = dataframe.get('dma') # calculate DMA
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['dataframe'] = dataframe.copy()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in dataframe.columns
        
    column_scaler = {}
    # scale the data (prices) from 0 to 1
    for column in feature_columns:
        scaler = preprocessing.MinMaxScaler()
        dataframe[column] = scaler.fit_transform(np.expand_dims(dataframe[column].values, axis=1))
        column_scaler[column] = scaler
    
    # add the MinMaxScaler instances to the result returned
    result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    dataframe['future'] = dataframe['adjclose'].shift(-n_days)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(dataframe[feature_columns].tail(n_days))
    
    # drop NaNs
    dataframe.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(dataframe[feature_columns].values, dataframe['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # se anade al resultado
    result['last_sequence'] = last_sequence
    
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    
    # Dividimos el resultado
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
    return result


def create_model(input_length, units, cell, num_layers, dropout,
                loss, normalizer,bidirectional=True,activation=activation):
    model = Sequential()
    if bidirectional == True :
        for i in range(num_layers):
            if i == 0:
                # first layer
                model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, input_length)))
               
            elif i == num_layers - 1:
                # last layer
                model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                model.add(Bidirectional(cell(units, return_sequences=True)))
                # add dropout after each laye
            model.add(Dropout(dropout))
    else:
        for i in range(num_layers):
            if i == 0:
                # first layer
                model.add(cell(units, return_sequences=True, input_shape=(None, input_length)))
               
            elif i == num_layers - 1:
                # last layer
                model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                model.add(cell(units, return_sequences=True))
                # add dropout after each laye
            model.add(Dropout(dropout))
    
    model.add(Dense(1, activation=activation))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=normalizer)

    return model
def build_model():
    model = Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model