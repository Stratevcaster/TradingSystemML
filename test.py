'''
Created on Feb 14, 2020

@author: USER
'''
from stock_prediction import create_model, load_data, np
from parameters import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy


def test(LOOKUP_STEP):
    preciosfutuos = np.array([])
    for step in range(1,LOOKUP_STEP):
        if step == 0:
            
            model_name = "{now}_{ticker_name}-{error_loss}-{cell_name}-seq-{sequence_lenght}-step-{step}-layers-{layers}-units-{neurons}".format(
                now=date_now,
                ticker_name=ticker,
                error_loss=LOSS,
                cell_name=CELL.__name__,
                sequence_lenght=N_STEPS,
                step=step,
                layers=N_LAYERS,
                neurons=UNITS
                )
            
        # load the data
            data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                         feature_columns=FEATURE_COLUMNS, shuffle=False)

        # construct the model
            model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER)

            model_path = os.path.join("results", model_name) + ".h5"
            model.load_weights(model_path)
    
            # evaluate the model
            mse, mae = model.evaluate(data["X_test"], data["y_test"])
            # calculate the mean absolute error (inverse scaling)
            mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
            print("Mean Absolute Error:", mean_absolute_error)
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
        elif step < LOOKUP_STEP and step< LOOKUP_STEP-1:
            model_name = "{now}_{ticker_name}-{error_loss}-{cell_name}-seq-{sequence_lenght}-step-{step}-layers-{layers}-units-{neurons}".format(
                now=date_now,
                ticker_name=ticker,
                error_loss=LOSS,
                cell_name=CELL.__name__,
                sequence_lenght=N_STEPS,
                step=step,
                layers=N_LAYERS,
                neurons=UNITS
                )
            
        # load the data
            data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                         feature_columns=FEATURE_COLUMNS, shuffle=False)

        # construct the model
            model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER)

            model_path = os.path.join("results", model_name) + ".h5"
            model.load_weights(model_path)
    
            # evaluate the model
            mse, mae = model.evaluate(data["X_test"], data["y_test"])
            # calculate the mean absolute error (inverse scaling)
            mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
            print("Mean Absolute Error:", mean_absolute_error)
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
            
            preciosfutuos=np.append(preciosfutuos,[predicted_price])
        elif step == LOOKUP_STEP-1:
            model_name = "{now}_{ticker_name}-{error_loss}-{cell_name}-seq-{sequence_lenght}-step-{step}-layers-{layers}-units-{neurons}".format(
                now=date_now,
                ticker_name=ticker,
                error_loss=LOSS,
                cell_name=CELL.__name__,
                sequence_lenght=N_STEPS,
                step=step,
                layers=N_LAYERS,
                neurons=UNITS
                )
            
        # load the data
            data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                         feature_columns=FEATURE_COLUMNS, shuffle=False)

        # construct the model
            model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER)

            model_path = os.path.join("results", model_name) + ".h5"
            model.load_weights(model_path)
    
            # evaluate the model
            mse, mae = model.evaluate(data["X_test"], data["y_test"])
            # calculate the mean absolute error (inverse scaling)
            mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
            print("Mean Absolute Error:", mean_absolute_error)
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
            
            preciosfutuos=np.append(preciosfutuos,[predicted_price])
            y_test = data["y_test"]
            X_test = data["X_test"]
            y_pred = model.predict(X_test)
            y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
            y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
            y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
            y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_pred[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
            
            accuracy_score(y_test, y_pred)
            
                    
            y_test = data["y_test"]
            X_test = data["X_test"]
              
            y_pred = model.predict(X_test)
            y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
            y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
            y_pred_new = numpy.append(y_pred, preciosfutuos)
            val = -1600 - len(preciosfutuos)
                
            plt.plot(y_test[-1600:], c='b')
            plt.plot(y_pred_new[val:], c='r')
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.legend(["Actual Price", "Predicted Price"])
            plt.show()
               
    return   preciosfutuos 

result=test(1)
print(result)
'''
    def predict(model, data, classification=False):
        # retrieve the last sequence from data
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
    return predicted_price
    '''



