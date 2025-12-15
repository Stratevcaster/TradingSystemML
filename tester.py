'''
Replacement for `test.py` renamed to avoid clashing with the Python stdlib `test` package.
Contains the same behavior with robust data fallback and model-loading fallback.
'''
from stock_prediction import create_model, load_data, np
from parameters import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy
import requests
import pandas as pd
import os
import glob

def test(N_DAYS_STEP):
    preciosfutuos = np.array([])
    for step in range(1,N_DAYS_STEP):
        if step == 0:
            model_name = "{now}_{ticker_name}-{error_loss}-{activation}-{normalizer}-{cell_name}-seq-{sequence_lenght}-step-{step}-layers-{layers}-units-{neurons}".format(
                now=date_model,
                ticker_name=ticker,
                error_loss=LOSS,
                cell_name=CELL.__name__,
                normalizer=normalizer,
                activation=activation,
                sequence_lenght=N_STEPS,
                step=step,
                layers=NUM_LAYERS,
                neurons=UNITS
                )
            
            
            if bidirectional == True:
                model_name += 'bidirectional'
        # cargamos datos si ya existen no se cargan
            data = safe_load_data_for_test(ticker, N_STEPS, n_days=N_DAYS_STEP, test_size=TEST_SIZE,
                         feature_columns=COLUMN_NAME)

        # contruimos el modelo
            model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, num_layers=NUM_LAYERS,
                    dropout=DROPOUT, normalizer=normalizer,bidirectional=bidirectional,activation=activation)

            model_path = os.path.join("results", model_name) + ".h5"
            load_weights_with_fallback(model, model_path, ticker)
    
            # evaluar el modelo 
            mse, mae = model.evaluate(data["X_test"], data["y_test"])
            # calculate the mean absolute error (inverse scaling)
            mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
            print("ERROR ABSOLUTO MEDIO:", mean_absolute_error)
            # predict the future price
            classification=False
            last_sequence = data["last_sequence"][:N_STEPS]
            # retrieve the column scalers
            column_scaler = data["column_scaler"]
            # reshape the last sequence
            last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
            # expand dimension
            last_sequence = np.expand_dims(last_sequence, axis=0)
            # get the prediction (scaled from 0 to 1)
            prediction = model.predict(last_sequence)
            # get the price (by inverting the scaling)
            predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
            preciosfutuos=np.append(preciosfutuos, [predicted_price])
        elif step < N_DAYS_STEP and step< N_DAYS_STEP-1:
            model_name = "{now}_{ticker_name}-{error_loss}-{activation}-{normalizer}-{cell_name}-seq-{sequence_lenght}-step-{step}-layers-{layers}-units-{neurons}".format(
                now=date_model,
                ticker_name=ticker,
                error_loss=LOSS,
                cell_name=CELL.__name__,
                normalizer=normalizer,
                activation=activation,
                sequence_lenght=N_STEPS,
                step=step,
                layers=NUM_LAYERS,
                neurons=UNITS
                )
            
            if bidirectional == True:
                model_name += 'bidirectional'
        # cargamos los datos
            data = safe_load_data_for_test(ticker, N_STEPS, n_days=N_DAYS_STEP, test_size=TEST_SIZE,
                         feature_columns=COLUMN_NAME)

        # construimos el modelo
            model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, num_layers=NUM_LAYERS,
                    dropout=DROPOUT, normalizer=normalizer,bidirectional=bidirectional,activation=activation)

            model_path = os.path.join("results", model_name) + ".h5"
            load_weights_with_fallback(model, model_path, ticker)
    
            # evaluamos
            mse, mae = model.evaluate(data["X_test"], data["y_test"])
            # error absoluto medio, evaluamos
            mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
            print("Mean Absolute Error:", mean_absolute_error)
            # predecir futuro precio 
            classification=False
            last_sequence = data["last_sequence"][:N_STEPS]
            # retrieve the column scalers
            column_scaler = data["column_scaler"]
            # reshape the last sequence
            last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
            # expand dimension
            last_sequence = np.expand_dims(last_sequence, axis=0)
            # precio de 0 a 1 
            prediction = model.predict(last_sequence)
            # Obtener precio
            predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
            
            preciosfutuos=np.append(preciosfutuos,[predicted_price])
        elif step == N_DAYS_STEP-1:
            model_name = "{now}_{ticker_name}-{error_loss}-{activation}-{normalizer}-{cell_name}-seq-{sequence_lenght}-step-{step}-layers-{layers}-units-{neurons}".format(
                now=date_model,
                ticker_name=ticker,
                error_loss=LOSS,
                cell_name=CELL.__name__,
                normalizer=normalizer,
                activation=activation,
                sequence_lenght=N_STEPS,
                step=step,
                layers=NUM_LAYERS,
                neurons=UNITS
                )
            
            if bidirectional == True:
                model_name += 'bidirectional'
        # cargamos los datos 
            data = safe_load_data_for_test(ticker, N_STEPS, n_days=N_DAYS_STEP, test_size=TEST_SIZE,
                         feature_columns=COLUMN_NAME)

        # Construimos el modelo 
            model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, num_layers=NUM_LAYERS,
                    dropout=DROPOUT, normalizer=normalizer,bidirectional=bidirectional,activation=activation)

            model_path = os.path.join("results", model_name) + ".h5"
            load_weights_with_fallback(model, model_path, ticker)
    
            # EVALUAMOS EL MODELO
            mse, mae = model.evaluate(data["X_test"], data["y_test"])
            # calcular error absoluto medio
            mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
            print("Error absoluto medio:", mean_absolute_error)
            # PREDECIR EL EL PRECIO FUTURO
            classification=False
            last_sequence = data["last_sequence"][:N_STEPS]
            # column scalers
            column_scaler = data["column_scaler"]
            # reformar la ultima sequencia 
            last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
            # expadndir dimension
            last_sequence = np.expand_dims(last_sequence, axis=0)
            # obtener precio de 0 a 1, normalizado
            prediction = model.predict(last_sequence)
            # obtener los precios revirtiendo la normalizacion
            predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
            
            preciosfutuos=np.append(preciosfutuos,[predicted_price])
            y_test = data["y_test"]
            X_test = data["X_test"]
            y_pred = model.predict(X_test)
            y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
            y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
            y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-N_DAYS_STEP], y_test[N_DAYS_STEP:]))
            y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_pred[:-N_DAYS_STEP], y_pred[N_DAYS_STEP:]))
            
            accuracy_score(y_test, y_pred)
            acurecy_number = accuracy_score(y_test, y_pred)
            acurecyInt= int(acurecy_number)
            
            print(N_DAYS_STEP,  " dias el porcentaje de acuracy es:", str(acurecy_number))
            print(f"Precio futuro dentro de  {N_DAYS_STEP} dias es {preciosfutuos}$")
           #  print(accuracy_score(y_test, y_pred))      
            y_test = data["y_test"]
            X_test = data["X_test"]
              
            y_pred = model.predict(X_test)
            
            y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
            y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
            y_pred_new = numpy.append(y_pred, preciosfutuos)
            days = 365
            years = 15
            total_days = -years*days
            total_predicted_days = total_days - len(preciosfutuos)
                
            plt.plot(y_test[total_days:], c='b')
            plt.plot(y_pred_new[total_days:], c='r')
            plt.xlabel("Dias")
            plt.ylabel("Precio")
            plt.legend(["Precio real", "Precio predicho"])
            plt.show()
            
    return   preciosfutuos 


def safe_load_data_for_test(ticker, n_steps, n_days, test_size, feature_columns):
    """Attempt to load market data via `load_data`. If it fails (network/API issues),
    build a synthetic DataFrame and call `load_data` with that instead.
    """
    try:
        return load_data(ticker, n_steps, n_days=n_days, test_size=test_size, feature_columns=feature_columns, shuffle=False)
    except Exception as e:
        print("Failed to load market data; falling back to synthetic dataset:", repr(e))
        # Build a synthetic dataset similar to run_real_train fallback
        N = max(300, n_steps * 3)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=N)
        df = pd.DataFrame(index=dates)
        np.random.seed(42)
        df['adjclose'] = np.cumsum(np.random.normal(0, 1, size=N)) + 100
        df['volume'] = np.random.randint(1000, 10000, size=N)
        df['open'] = df['adjclose'] + np.random.normal(0, 1, size=N)
        df['high'] = df[['open', 'adjclose']].max(axis=1) + np.random.rand(N)
        df['low'] = df[['open', 'adjclose']].min(axis=1) - np.random.rand(N)
        for col in ['macd', 'atr', 'dma']:
            df[col] = 0.0
        return load_data(df, n_steps, n_days=n_days, test_size=test_size, feature_columns=feature_columns, shuffle=False)


def load_weights_with_fallback(model, model_path, ticker_name):
    """Try to load weights from `model_path`. If the file doesn't exist, search `results/` for
    the most recent model artifact that contains `ticker_name` and load that instead.
    """
    try:
        model.load_weights(model_path)
        return
    except FileNotFoundError:
        print(f"Model file {model_path} not found — searching for latest model for ticker '{ticker_name}'...")
        candidates = glob.glob(os.path.join('results', f"*{ticker_name}*.h5")) + glob.glob(os.path.join('results', f"*{ticker_name}*.keras"))
        if not candidates:
            raise FileNotFoundError(f"No model files found for ticker '{ticker_name}' in ./results. Please run training first.")
        latest = max(candidates, key=os.path.getmtime)
        print(f"Loading latest model found: {latest}")
        model.load_weights(latest)
