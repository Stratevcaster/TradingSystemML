'''
Created on Feb 14, 2020

@author: Yani STRATEV
'''
from stock_prediction import create_model, load_data, np
from parameters import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy
import requests

def test(N_DAYS_STEP):
    preciosfutuos = np.array([])
    for step in range(1,N_DAYS_STEP):
        if step == 0:
        
            model_name = "{date_model}_{ticker_name}-{error_loss}-{cell_name}-seq-{sequence_lenght}-step-{step}-layers-{layers}-units-{neurons}".format(
                now=date_now,
                date_model = date_model,
                ticker_name=ticker,
                error_loss=LOSS,
                cell_name=CELL.__name__,
                sequence_lenght=N_STEPS,
                step=step,
                layers=NUM_LAYERS,
                neurons=UNITS
                )
            if bidirectional == True:
                model_name += 'bidirectional'
        # cargamos datos si ya existen no se cargan
            data = load_data(ticker, N_STEPS, n_days=N_DAYS_STEP, test_size=TEST_SIZE,
                         feature_columns=COLUMN_NAME, shuffle=False)

        # contruimos el modelo
            model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, num_layers=NUM_LAYERS,
                    dropout=DROPOUT, normalizer=normalizer,bidirectional=False)

            model_path = os.path.join("results", model_name) + ".h5"
            model.load_weights(model_path)
    
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
            model_name = "{date_model}_{ticker_name}-{error_loss}-{cell_name}-seq-{sequence_lenght}-step-{step}-layers-{layers}-units-{neurons}".format(
                now=date_now,
                date_model = date_model,
                ticker_name=ticker,
                error_loss=LOSS,
                cell_name=CELL.__name__,
                sequence_lenght=N_STEPS,
                step=step,
                layers=NUM_LAYERS,
                neurons=UNITS
                )
            if bidirectional == True:
                model_name += 'bidirectional'
        # cargamos los datos
            data = load_data(ticker, N_STEPS, n_days=N_DAYS_STEP, test_size=TEST_SIZE,
                         feature_columns=COLUMN_NAME, shuffle=False)

        # construimos el modelo
            model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, num_layers=NUM_LAYERS,
                    dropout=DROPOUT, normalizer=normalizer,bidirectional=bidirectional)

            model_path = os.path.join("results", model_name) + ".h5"
            model.load_weights(model_path)
    
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
            model_name = "{date_model}_{ticker_name}-{error_loss}-{cell_name}-seq-{sequence_lenght}-step-{step}-layers-{layers}-units-{neurons}".format(
                now=date_now,
                ticker_name=ticker,
                date_model = date_model,
                error_loss=LOSS,
                cell_name=CELL.__name__,
                sequence_lenght=N_STEPS,
                step=step,
                layers=NUM_LAYERS,
                neurons=UNITS
                )
            if bidirectional == True:
                model_name += 'bidirectional'
        # cargamos los datos 
            data = load_data(ticker, N_STEPS, n_days=N_DAYS_STEP, test_size=TEST_SIZE,
                         feature_columns=COLUMN_NAME, shuffle=False)

        # Construimos el modelo 
            model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, num_layers=NUM_LAYERS,
                    dropout=DROPOUT, normalizer=normalizer,bidirectional=bidirectional)

            model_path = os.path.join("results", model_name) + ".h5"
            model.load_weights(model_path)
    
            # EVALUAMOS EL MODELO
            mse, mae = model.evaluate(data["X_test"], data["y_test"])
            # calcular error absoluto medio
            mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
            print("Mean Absolute Error:", mean_absolute_error)
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






